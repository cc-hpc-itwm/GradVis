# coding: utf-8
#import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

def vprint(*vargs, **kwargs):
    if vprint.verbosity >0:
        print(*vargs, **kwargs)
vprint.verbosity=0


def vectoriz(vector_orig,parameter):
    """
    Reshapes vector into model shape.

    Args:
        vector_orig: unstructured array
        parameter: python array of numpy arrays (target shape)
    Return:
        python array of numpy arrays, with data from vecotr_orig and shape like parameter
    """
    vector = []
    indic = 0
    #zahler = 0
    for p in parameter:
        len_p = p.size
        p_size = p.shape
        vec_it = vector_orig[indic:(indic+len_p)].reshape(p_size)#[p_size[i] for i in range(len(p_size))])
        #vector.append(torch.tensor(vec_it, dtype=torch.float32))
        vector.append(np.array(vec_it, dtype=np.float32))
        indic += len_p
        #zahler += 1
    return vector


def get_params(parameter):
    """
    Concatenates a python array of numpy arrays into a single, flat numpy array.
    """
    return np.concatenate( [ar.flatten() for ar in parameter], axis=None)


def mask_layers(parameter,layername):
    new_pars = parameter.copy()
    i=0

    for key,val in new_pars.items():
        if layername in key:
            continue
        else:
            val*=0
        i+=1
    return new_pars



def get_pca_vec(model, filenames, layer_names, pca_direcs=None):
    """
    Calculates the principal components of the model parameters.
    Does not modify the internal state of model. Does not execute model.

    Args:
        model: nn model, with nn_model.Base_NNModel interface
        filenames: filenames of the checkpoints which shall be included into the PCA.
        pca_direcs: array of PCA directions to be computed
    Return:
        Two vectors with highest variance
    """
    mats = []

    for file in filenames:
        testi = model.get_parameters(file)
        parlis = np.ndarray([0])

        if layer_names is not None:
            for k in range(len(layer_names)):
                testi_temp = mask_layers(testi,layer_names[k])
                if k==0:
                    testi = testi_temp
                else:
                    testi += testi_temp



        for key in testi:
            if "weight" not in key and "bias" not in key:
                testi[key] *= 0
            parlis = np.concatenate((parlis,testi[key]), axis=None)
        pas = parlis
        #pas = get_params(parlis)
        mats.append(pas)
    mats = np.vstack(mats)
    mats_new = mats[:-1]-mats[-1]

    data = mats_new

    if pca_direcs is not None:
        if len(pca_direcs) != 2:
            raise ValueError("Expected pca_direcs to be array of length 2, got {}!".format(len(pca_direcs)))
        max_comp = max(pca_direcs)
        pca = PCA(n_components=max_comp)
        principalComponents = pca.fit_transform(data.T)
        vprint("Principal",pca.explained_variance_ratio_)

        variance_ratio = [pca.explained_variance_ratio_[pca_direcs[0]-1],pca.explained_variance_ratio_[pca_direcs[1]-1]]
        prince_comp1 = np.array(principalComponents[:,pca_direcs[0]-1])
        prince_comp2 = np.array(principalComponents[:,pca_direcs[1]-1])

        return prince_comp1, prince_comp2, variance_ratio

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data.T)
    vprint("Principal",pca.explained_variance_ratio_)


    return np.array(principalComponents[:,0]),np.array(principalComponents[:,1]),pca.explained_variance_ratio_


def cont_loss(model,parameter,alph,bet,get_v,get_w):
    """
    Calculates the loss landscape based on vectors v and w (which can be principal components).
    Changes the internal state of model. Executes model.

    Args:
        model: nn model, with nn_model.Base_NNModel interface
        parameter: weights of the converged net, centered point of analysis
        alph: list of scalars for 1st direction
        bet: scalar for 2nd direction
        get_v: 1st direction
        get_w: 2nd direction
    Return:
        list of loss values
    """
    vals = []
    for al in alph:
        testi_clone = parameter.copy()
        ind = 0
        # calculate new parameters for model
        for key in parameter:
            #testi_clone[key] = testi_clone[key].cpu().detach() + al*get_v[ind] + bet*get_w[ind]
            testi_clone[key] = testi_clone[key] + al*get_v[ind] + bet*get_w[ind]
            ind += 1

        # load parameters into model and calcualte loss
        model.set_parameters(testi_clone)
        loss = model.calc_loss()
        vals = np.append(vals,loss)
    return vals


def give_coefs(model, filenames, parameter, v, w,layername=None):
    """
    Calculates the scale factors for plotting points in the 2D space spanned by the vectors v and w.

    Args:
        model: nn model, with nn_model.Base_NNModel interface
        filenames: checkpoint files, which define the trajectory.
        parameter: central point, to which the trajectory will be calculated.
        v: 1st vector spanning the 2D space
        w: 2nd vector spanning the 2D space
    Return:
        list of coefficients
    """
    #par_step = torch_params_to_numpy(torch.load(filename+str(i)))
    matris = [v,w]
    matris= np.vstack(matris)
    matris = matris.T

    if layername is not None:
        for k in range(len(layername)):
            testi_temp = mask_layers(parameter,layername[k])
            if k==0:
                parameter = testi_temp
            else:
                parameter += testi_temp

    parlis = parameter.values()
    pas = get_params(parlis)
    coefs = []
    for file in filenames:
        par_step = model.get_parameters(file)

        if layername is not None:
            for k in range(len(layername)):
                testi_temp = mask_layers(par_step,layername[k])
                if k==0:
                    par_step = testi_temp
                else:
                    par_step += testi_temp

        parstep = par_step.values()
        st = get_params(parstep)

        b = st-pas
        coefs.append(np.hstack(np.linalg.lstsq(matris,b,rcond=None)[0]))

    return(coefs)

def normalize(parameter,get_v,get_w):
    """
    Normalizes the vectors spanning the 2D space, to make trajectories comparable between each other.

    Args:
        parameter: the parameters to normalize to.
        get_v, get_w: the vectors in the 2D space, which should be normalized to 'parameter'.
    Return:
        tuple of normalized vectors get_v, get_w
    """

    parlis = list(parameter.values())

    for i in range(len(parlis)):
        if "weight" in list(parameter.keys())[i] or "bias" in list(parameter.keys())[i]:
            factor_v = np.linalg.norm(parlis[i])/np.linalg.norm(get_v[i])
            factor_w = np.linalg.norm(parlis[i])/np.linalg.norm(get_w[i])
            get_v[i] = get_v[i]*factor_v
            get_w[i] = get_w[i]*factor_w
        else:
            get_v[i] = get_v[i]*0
            get_w[i] = get_w[i]*0

    return get_v,get_w

################################
### Main function
################################
def _visualize(model,filenames,N,random_dir=False,proz=0.5,v_vec=[],w_vec=[],verbose=False,layername=None,pca_dirs=None):
    """
    Main function to visualize trajectory in parameterspace.

    Args:
        model: nn model, with nn_model.Base_NNModel interface
        filenames: list of checkpoint names (files with parameters), orderered with the centerpoint last in list
        N: number of grid points for plotting (for 1 dim)
        random_dir (bool): if random directions should be used instead of PCA
        proz: margins for visualized space (in %)
        v_vec, w_vec: if defined, custom vectors will be used instead of PCA
        verbose: verbosity of prints
        pca_dirs: choose the pca directions to be plotted, if None the first two are chosen
    Return:
        Array containing loss values, path values, variance data and the two pca components. Also a flag value.
    """
    # verbosity level settings
    if verbose:
        vprint.verbosity=1
    progress_bar_wrapper = None
    if verbose:
        progress_bar_wrapper = lambda x : tqdm(x)
    else:
        progress_bar_wrapper = lambda x : x


    # load the parameters of the final step
    parameter = model.get_parameters()#filename=filenames[-1])
    parlis = list(parameter.values()) # list of 'parameter' values

    if random_dir:
        vprint("Calculating directions...")
    else:
        vprint("Calculating PCA directions...")

    if len(v_vec)!=0 and len(w_vec)!=0:
        v = v_vec
        w = w_vec
    elif random_dir:
        pytorch_total_params = sum(p.numel() for p in list(parlis))
        v = np.random.normal(size=pytorch_total_params)
        w = np.random.normal(size=pytorch_total_params)
    else:
        v,w,pca_variance = get_pca_vec(model, filenames,layername,pca_direcs=pca_dirs)

    get_v = vectoriz(v,parlis)
    get_w = vectoriz(w,parlis)

    vprint("Normalizing...")
    get_v,get_w = normalize(parameter, get_v, get_w)

    if not random_dir:
        vprint("Calculating coefficients...")

        v = get_params(get_v)
        w = get_params(get_w)

        coefs = give_coefs(model, filenames, parameter, v, w,layername)
        coefs = np.array(coefs)
        paths = []


        if layername is not None:
            for k in range(len(layername)):
                testi_temp = mask_layers(parameter,layername[k])
                if k==0:
                    parameter = testi_temp
                else:
                    parameter += testi_temp


        vprint("Calculating Z-values of paths...")
        for val in progress_bar_wrapper(range(len(coefs))):
            yo = cont_loss(model, parameter,[coefs[val][0]],coefs[val][1],get_v,get_w)
            paths.append(yo)

        paths = np.array(paths)
        coefs_x = coefs[:,0][np.newaxis]
        coefs_y = coefs[:,1][np.newaxis]


        n = N
        proz = proz
        boundaries_x = max(coefs_x[0])-min(coefs_x[0])
        boundaries_y = max(coefs_y[0])-min(coefs_y[0])

        x = np.linspace(min(coefs_x[0])-proz*boundaries_x, max(coefs_x[0])+proz*boundaries_x, n)
        y = np.linspace(min(coefs_y[0])-proz*boundaries_y, max(coefs_y[0])+proz*boundaries_y, n)

    else:
        n = N
        proz = proz
        boundaries_x = 5.
        boundaries_y = 5.

        x = np.linspace(-proz*boundaries_x, proz*boundaries_x, n)
        y = np.linspace(-proz*boundaries_y, proz*boundaries_y, n)


    X, Y = np.meshgrid(x, y)


    vprint("Calculating loss landscape...")

    Z = []

    for i in progress_bar_wrapper(range(len(y))):
        vals = cont_loss(model,parameter,X[i],Y[i][0],get_v,get_w)
        Z.append(vals)


    if not random_dir:
        if len(v_vec)!=0 and len(w_vec)!=0:
            return [(X,Y,np.vstack(Z)),(coefs_x[0],coefs_y[0],paths.T[0])],1
        else:
            cache = (pca_variance,v,w)
            return [(X,Y,np.vstack(Z)),(coefs_x[0],coefs_y[0],paths.T[0]),cache],2
    else:
        cache = (v,w)
        return [(X,Y,np.vstack(Z)),cache],3



def visualize(model,filenames,N,path_to_file,random_dir=False,proz=0.5,v_vec=[],w_vec=[],verbose=False,layername=None,pca_dirs=None):
    """
    Wrapper for _visualize function that saves results as npz (numpy_compressed) file

    Args:
        model: nn model, with nn_model.Base_NNModel interface
        filenames: list of checkpoint names (files with parameters), orderered with the centerpoint last in list
        N: number of grid points for plotting (for 1 dim)
        path_to_file: path and filename where the results are going to be saved at
        random_dir (bool): if random directions should be used instead of PCA
        proz: margins for visualized space (in %)
        v_vec, w_vec: if defined, custom vectors will be used instead of PCA
        verbose: verbosity of prints
        pca_dirs: choose the pca directions to be plotted, if None the first two are chosen
    """

    my_file = Path(path_to_file+".npz")

    if my_file.is_file():
        print("File {} already exists!".format(path_to_file+".npz"))
    else:
        outputs,flag = _visualize(model,filenames,N,random_dir=random_dir,proz=proz,v_vec=v_vec,w_vec=w_vec,verbose=verbose,layername=layername,pca_dirs=pca_dirs)
        np.savez_compressed(path_to_file, a=outputs, b=flag)