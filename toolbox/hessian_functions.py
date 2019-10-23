import torch
import numpy as np
import torch.autograd as autograd
from scipy import sparse
from scipy.sparse import linalg
from tqdm import tqdm
import scipy
import imageio

def vprint(*vargs, **kwargs):
    if vprint.verbosity >0:
        print(*vargs, **kwargs)
vprint.verbosity=0

def get_eigenvector(model,dataloader,criterion,filename,num_eigs=2, use_gpu=True, percentage=0.05, num_iters=1, mode='LA'):
    """
    Compute Eigenvectors of the Hessian and save them to the hard drive

    Args:
        model: nn model, with nn_model.Base_NNModel interface
        dataloader: dataloader in Pytorch in order to get access to samples
        criterion: Loss function
        filename: path and filename where the resulting eigenvectors are going to be saved at
        num_eigs: Number of eigenvectors and eigenvalues to compute
        max_samples: Number of samples to choose
        use_gpu (bool): Mode to use the GPU for the calculations
        mode: Which eigenvalues to compute (Largest magnitude, etc..)
    """
    num_pars = Num_pars_origs(model)
    opi = HessVec(model, dataloader, criterion, use_gpu=use_gpu, percentage=percentage,num_iters=num_iters)
    A = linalg.LinearOperator((num_pars,num_pars), matvec=lambda v: opi.apply(torch.tensor(v).float()))

    vals, vecs = sparse.linalg.eigsh(A, k=num_eigs, which=mode)

    print("Eigenvalues are {:.2f} and {:.2f}".format(vals[0],vals[1]))

    all_vecs = []

    for l in range(num_eigs):
        if next(model.parameters()).is_cuda:
            new_state = model.cpu().state_dict()
            model = model.cuda()
        else:
            new_state = model.state_dict()

        indic=0
        for k,j in new_state.items():
            len_p = j.numpy().size
            p_size = j.numpy().shape
            if "running" in k or "batch" in k:
                new_state[k] = new_state[k].numpy()
            else:
                new_state[k] = vecs[indic:(indic+len_p),l].reshape(p_size)
                indic += len_p

        new_vec = []
        for i,k in enumerate(new_state.keys()):
            if i==0:
                new_vec = new_state[k].ravel()
            else:
                new_vec = np.concatenate((new_vec,new_state[k].ravel()))

        all_vecs.append(new_vec)

    np.save(filename+"_vecs",np.vstack(all_vecs))
    np.save(filename+"_vals",vals)




def gaussian(lam, t, sig):
    """
    Gaussian function used to approximate the Eigenvalue density spectrum

    Args:
        lam: offset in Gaussian
        t: x-value of the Gaussian
        sig: Sigma value of Gaussian
    """
    return 1./(sig*np.sqrt(2.*np.pi))*np.exp(-np.power(t - lam, 2.) / (2 * np.power(sig, 2.)))


def phi_comp(x,nodes,deco_weights,sigma):
    """
    Computes the Hessian Eigenvalue density spectrum given the Eigenvalues and Eigenvectors

    Args:
        x: x-value in plot
        nodes: Eigenvalues, corresponding to position of specific Gaussian on the spectrum
        deco_weights: Determine the height of the peaks of each Gaussian
        sigma: Sigma value of Gaussian
    """
    phi = deco_weights[0]*gaussian(nodes[0],x,sigma)
    for i in range(1,len(nodes)):
        phi_t = deco_weights[i]*gaussian(nodes[i],x,sigma)
        phi += phi_t
    return phi



def Num_pars_origs(model):
    '''
    Returns the number of weights in a Neural Network

    Args:
        model: nn model, with nn_model.Base_NNModel interface
    '''
    num_pars = 0
    for k,param in model.named_parameters():
        p_len = param.numel()
        num_pars += p_len
    return num_pars


def lanczos_with_q_torch(A,psi,N):
    """
    Lanczos with reorthogonalization, returns the alpha and beta values of the resulting tridiagonal matrix

    Args:
        A: Function returning the Hessian-vector product
        psi: Vector of the Hessian-vector product
        N: Number of iteration, corresponding to number of Eigenvalues to be computed
    """
    qs = []
    dim=len(psi)
    psi = psi[:,None]
    q0 = torch.zeros((dim,1),dtype=psi[0].dtype)
    q1 = psi/torch.norm(psi,p=2)
    qs.append(q0)
    qs.append(q1)

    Q = torch.clone(q1).reshape(-1, 1)

    a = []
    b = []
    b.append(torch.norm(psi,p=2))
    beta = 0.0
    for k in range(N):

        v = A(q1.squeeze(1).float()).double()[:,None].cpu() - beta*q0
        alpha = torch.mm(q1.t(),v)[0]

        v = v - alpha*q1
        v -= torch.mm(Q,torch.mm(Q.t(),v))

        beta = torch.norm(v,p=2)

        q0 = q1
        q1 = v/beta

        Q = torch.cat((Q, q1.reshape(-1, 1)), 1)

        a.append(alpha)
        if k < N-1:
            b.append(beta)
        if torch.norm(v,p=2) < 1e-13:
            a = torch.stack(a)
            b = torch.stack(b)
            return a.cpu().numpy(),b.cpu().numpy()

    a = torch.stack(a)
    b = torch.stack(b)
    return a.cpu().numpy(),b.cpu().numpy()


def phi_comp_mult(x,all_eigs,all_vecs,sigma_sq):
    """
    Caluculates the Hessian Eigenvalue spectrum of all runs together

    Args:
        x: x-value in plot
        all_eigs: Eigenvalues, corresponding to position of specific Gaussian on the spectrum
        all_vecs: Eigenvectors that correspond to the height of the peaks of each Gaussian
        sigma_sq: Sigma value of Gaussian
    """
    phi_finalo = np.zeros(len(x))
    len_eigs = len(all_eigs)
    for i in range(len_eigs):
        phi_finalo += phi_comp(x,all_eigs[i],all_vecs[i],sigma_sq)

    return phi_finalo


def stochastic_lanczos(model,dataloader,criterion,filename,num_repeats=10,num_eigs=80, percentage=0.2, use_gpu=True,num_iters=1,verbose=False):
    """
    Stochastic Lanczos Eigenvalue computations, saves all the computed Eigenvalues and Eigenvectors and returns a function in order to plot the spectrum.

    Args:
        model: nn model, with nn_model.Base_NNModel interface
        dataloader: dataloader in Pytorch in order to get access to samples
        criterion: Loss function
        filename: path and filename where the resulting eigenvectors are going to be saved at
        num_repeats: number of iterations used in the Algorithm
        num_eigs: Number of eigenvectors and eigenvalues to compute
        percentage: percentage of samples to use
        use_gpu (bool): Mode to use the GPU for the calculations
        num_iters: Number of batches to accumulate before calculating the Hessian-vector product
    """

    if verbose:
        vprint.verbosity=1
    progress_bar_wrapper = None
    if verbose:
        progress_bar_wrapper = lambda x : tqdm(x)
    else:
        progress_bar_wrapper = lambda x : x

    num_pars = Num_pars_origs(model)

    vprint("Number of parameters in the network are: {}".format(num_pars))

    opi = HessVec(model, dataloader, criterion, use_gpu=use_gpu, percentage=percentage,num_iters=num_iters)

    all_eigenvals = []
    all_eigenvecs = []
    for _ in progress_bar_wrapper(range(num_repeats)):
        v_init = torch.tensor(np.random.normal(0,1/float(num_pars),num_pars))

        al,bet = lanczos_with_q_torch(lambda x: opi.apply(x),v_init,num_eigs)
        eigi,veci = scipy.linalg.eigh_tridiagonal(al.T[0],bet[1:])
        all_eigenvals.append(eigi)
        all_eigenvecs.append(veci[0,:]**2)

    vprint("Saving calculated values...")

    np.savez_compressed(filename, a=all_eigenvals, b=all_eigenvecs,c=1.0/num_repeats)

    def func(x,sigma_sq):
        return 1.0/num_repeats*phi_comp_mult(x,all_eigenvals,all_eigenvecs,sigma_sq)

    return func


def get_xlim(filearray):
    """
    Returns the lower and upper limit of the x-axis for a given array filenames

    Args:
        filearray: array of filenames generated by the stochastic Lanczos quadrature
    """
    realmax = 0
    realmin = 0
    for name in filearray:
        outs = np.load(name)
        outs = outs["a"]
        maxval = np.amax(outs)
        minval = np.amin(outs)

        if maxval >= realmax:
            realmax = maxval
        if minval <= realmin:
            realmin = minval

    return minval,maxval


def get_spectrum(x,sigma_sq,filename):
    """
    Returns the y-values of a given file

    Args:
        x: numpy array of x-values for plotting
        sigma_sq: sigma value of the Gaussians used to approximate the spectrum
        filename: name of the file generated by the stochastic Lanczos quadrature algorithm
    """

    file = np.load(filename)
    eigenvals = file["a"]
    eigenvecs = file["b"]
    factor = file["c"]

    return factor*phi_comp_mult(x,eigenvals,eigenvecs,sigma_sq)


def get_gif(filenames,savefile,fps=2):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(savefile+'.gif', images,fps=fps)


class Operator:
    """
    maps x -> Lx for a linear operator L
    """

    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError

class HessVec(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_samples: max number of examples per batch using all GPUs.
    """

    def __init__(self, model, dataloader, criterion, use_gpu=True,
                 percentage=0.2,num_iters=1):
        size = int(sum(p.numel() for p in model.parameters()))
        super(HessVec, self).__init__(size)
        self.grad_vec = torch.zeros(size)
        self.model = model
        assert percentage <= 1
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.percentage = percentage
        self.num_iters = num_iters

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        full_hessian = None
        # compute original gradient, tracking computation graph
        if self.use_gpu:
            vec = vec.cuda()
        self.zero_grad()

        grad_vec = None
        self.dataloader_iter = iter(self.dataloader)
        n = int(np.ceil(len(self.dataloader)*self.percentage))

        for k in range(n):
            batch_grad = self.prepare_grad()
            if grad_vec is None:
                grad_vec = batch_grad
            else:
                grad_vec += batch_grad

            if k%self.num_iters >= self.num_iters-1:
                grad_vec /= self.num_iters
                self.zero_grad()

                # take the second gradient
                grad_grad = torch.autograd.grad(grad_vec, self.model.parameters(),
                                            grad_outputs=vec,
                                            only_inputs=True)
                # concatenate the results over the different components of the network
                hessian_vec_prod = torch.cat([g.contiguous().view(-1)
                                        for g in grad_grad])
                if full_hessian is not None:
                    full_hessian += hessian_vec_prod
                else:
                    full_hessian = hessian_vec_prod
                grad_vec = None

        if n%self.num_iters != 0:
            grad_vec /= (n%self.num_iters)
            self.zero_grad()

            # take the second gradient
            grad_grad = torch.autograd.grad(grad_vec, self.model.parameters(),
                                        grad_outputs=vec,
                                            only_inputs=True)
            # concatenate the results over the different components of the network
            hessian_vec_prod = torch.cat([g.contiguous().view(-1)
                                        for g in grad_grad])
            if full_hessian is not None:
                full_hessian = self.num_iters/n*full_hessian+(n%self.num_iters)/n*hessian_vec_prod
            else:
                full_hessian = hessian_vec_prod

        return full_hessian.cpu()

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            all_inputs, all_targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            all_inputs, all_targets = next(self.dataloader_iter)

        num_chunks = 1
        grad_vec = None

        if self.use_gpu:
            input = all_inputs.cuda()
            target = all_targets.cuda()

        output = self.model(input)
        loss = self.criterion(output, target)
        grad_dict = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
        self.grad_vec = grad_vec
        return self.grad_vec