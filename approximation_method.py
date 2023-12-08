import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global select_number
select_number = 5

class softmax_approximation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        global select_number
        ctx.constant = select_number
        S = ctx.constant   
        tensor = F.softmax(input, dim=-1)
        m = torch.full(tensor.shape[:-1] + (2*S+1,), -1e1, device=device)

        if mask is not None:
            inner_tensor, index = torch.masked_fill(tensor, ~mask, -10).sort(descending=True)
            index = torch.argsort(index).to(dtype=torch.uint8)
            
            # save high value
            m[:,:,:,:S] = (torch.topk(inner_tensor,  S)[0]) + 1e-8
            # save low value
            m[:,:,:,-S-1:-1] = (torch.flip(-1*(torch.topk(torch.masked_fill(-inner_tensor, ~mask, -10), S)[0]), dims=(-1,))) + 1e-8

            idx = torch.nonzero(torch.sum(mask, dim=-1) <= 2*S)
            m[idx[:, 0], :, :, :2*S] = inner_tensor[idx[:, 0], :, :, :2*S] + 1e-8
    
            del idx

        else:
            inner_tensor, index = tensor.sort(descending=True)
            index = torch.argsort(index).to(dtype=torch.uint8)

            # save high value
            m[:,:,:,:S] = inner_tensor[...,:S] + 1e-8
            # save low value
            m[:,:,:,-S-1:-1] = inner_tensor[...,-S:] + 1e-8


        #[beta = (1 / (high * low)).sqrt() * (high - gamma).exp()]
        beta_hat = torch.sqrt(torch.nan_to_num(torch.div(1,torch.mul(m[..., S-1, None], m[..., -S-1, None]))))* \
                (torch.exp(m[..., S-1, None]-inner_tensor[...,S, None]))
                                    
        beta_hat = torch.nan_to_num(beta_hat)

        #[y = (-1 / (high**2 * beta**2)).log() * (-1 / beta)]
        sigma = torch.log(torch.nan_to_num(torch.div(1,torch.mul((m[..., S-1, None])**2, beta_hat**2))))* \
                (-1*torch.nan_to_num(torch.div(1,beta_hat)))
        sigma = torch.nan_to_num(sigma)
        # 1 / (1-(1-h_m)) * y * beta

        if mask is not None:
                ss = torch.nan_to_num(torch.div(1,(mask.sum(-1)-2*(S-1)-1).unsqueeze(-1)))
                m[:,:,:,-1] = torch.mul(ss,(torch.mul(beta_hat, torch.mul((sigma),torch.nan_to_num(torch.div(1,1-(m[..., S-1, None]))))))).squeeze(-1)
                del ss

        else:
                m[:,:,:,-1] = torch.mul(torch.div(1, index.shape[-1]),(torch.mul(beta_hat, torch.mul((sigma),torch.nan_to_num(torch.div(1,1-(m[..., S-1, None]))))))).squeeze(-1)

        m = torch.cat((m, torch.exp(tensor).sum(dim=-1).unsqueeze(-1)), dim =-1)
        
        ctx.save_for_backward(m, index, mask)
        del m, inner_tensor, index, beta_hat, sigma
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        S = ctx.constant

        m, index, mask = ctx.saved_tensors

        if mask is not None:
            approx_mask = mask.clone()
            approx_mask[(mask.cumsum(dim=-1)) > (mask.cumsum(dim=-1)).gather(-1, (mask.cumsum(dim=-1)).max(-1, keepdim=True)[1]) - S] = False
            approx_mask[...,:S] = False

            select_mask = mask * ~approx_mask


            temp_matrix = torch.zeros_like(index).to(dtype=torch.float32)
            temp_matrix = temp_matrix.masked_scatter(select_mask, m[...,:-2][m[...,:-2] != -1e1])

            arg_value = torch.arange(index.shape[-1] - 2* (S - 1), device=mask.device).unsqueeze(0).expand(index.shape[:-1] + (-1,))

            approx_value = torch.mul(m[...,:,S-1].unsqueeze(-1), torch.exp(-1*torch.mul(m[...,-2, None], arg_value))).to(device)
            approx_value = torch.cat((torch.zeros(index.shape[:-1] + (S-1,), device= mask.device), approx_value, torch.zeros(index.shape[:-1] + (S-1,), device= mask.device)), dim = -1)

            approx_value = approx_value.masked_fill(~approx_mask, 0)
            approx_value = torch.mul(approx_value, (1-(temp_matrix.sum(-1).unsqueeze(-1)))/approx_value.sum(-1).unsqueeze(-1))

            approx_value = approx_value.masked_fill(~approx_mask, -1e1)
            temp_matrix = temp_matrix.masked_scatter(approx_mask, approx_value[approx_value != -1e1])

            temp_matrix = torch.gather(temp_matrix, dim=-1, index=index.type(dtype=torch.int64))
            temp_matrix = temp_matrix.masked_fill(mask==0, 0)

            del approx_mask, select_mask, arg_value, approx_value


        else: 
            idx = torch.arange(index.shape[-1]-(2*(S-1))).expand(m.shape[0], m.shape[1], m.shape[2], index.shape[-1]-(2*(S-1))).to(device)
            temp_matrix = torch.mul(m[...,:,S-1].unsqueeze(-1), torch.exp(-1*torch.mul(m[...,-2, None], idx))).to(device)
            temp_matrix[...,1:-1] = torch.mul(temp_matrix[...,1:-1], ((1-m[...,:-2].sum(-1).unsqueeze(-1)) / temp_matrix[...,1:-1].sum(-1).unsqueeze(-1)))
            temp_matrix = torch.cat((m[...,:,:S], temp_matrix[...,1:-1], m[...,:,S:-2]),dim=-1)
            temp_matrix = torch.gather(temp_matrix, dim=-1, index=index.type(dtype=torch.int64))
            
            del idx

        grad_matrix = torch.diag_embed(temp_matrix) - torch.einsum('...j, ...i -> ...ji', temp_matrix, temp_matrix)

        del m, index, mask, temp_matrix
        
        return torch.einsum("...i, ...ij", grad_output, grad_matrix), None