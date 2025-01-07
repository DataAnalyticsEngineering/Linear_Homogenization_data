import numpy as np
"""
Utility routines to work with tensors
in Voigt and Mandel notation.

For the notation see also the jupyter notebook.
"""

def VoigtStrain2Mandel( A_voigt, order = 'voigt' ):
    """Convert a strain in Voigt notation to Mandel notation.
    
    Parameters
    ----------
    A_voigt : ndarray
        symmetric 2-tensor in Voigt-like notation
        (i.e. engineering shear for off-diagonal components)

    order : str
        if 'voigt' the order (xx, yy, zz, yz, xz, xy) and
        otherwise, (xx, yy, zz, xy, xz, yz)  is assumed;

    Returns
    -------
    ndarray
        The converted tensor as a 6 vector

    """
    f = np.sqrt(0.5)
    A_mandel = np.array([1., 1., 1., f, f, f])*A_voigt
    if(order == 'voigt'):
        # Voigt in --> Reordering needed
        A_mandel = A_mandel[np.array((0, 1, 2, 5, 4, 3))]
    return A_mandel

def VoigtStress2Mandel( A_voigt, order = 'voigt' ):
    """Convert a stress in Voigt notation to Mandel notation.
    
    Parameters
    ----------
    A_voigt : ndarray
        symmetric 2-tensor in Voigt-like notation
        (i.e. off-diagonal components are reported without prefactor)

    order : str
        if 'voigt' the order (xx, yy, zz, yz, xz, xy) and
        otherwise, (xx, yy, zz, xy, xz, yz)  is assumed;

    Returns
    -------
    ndarray
        The converted tensor as a 6 vector

    """

    f = np.sqrt(2.0)
    A_mandel = np.array([1., 1., 1., f, f, f])*A_voigt
    if(order == 'voigt'):
        # Voigt in --> Reordering needed
        A_mandel = A_mandel[np.array((0, 1, 2, 5, 4, 3))]
    return A_mandel

def Mandel2VoigtStrain( A_mandel, order = 'voigt' ):
    """Convert a tensor in Mandel notation to Voigt (for strains).
    
    Parameters
    ----------
    A_voigt : ndarray
        symmetric 2-tensor in Mandel notation

    order : str
        if 'voigt' the order (xx, yy, zz, yz, xz, xy) and
        otherwise, (xx, yy, zz, xy, xz, yz)  is returned on output;

    Returns
    -------
    ndarray
        The converted tensor as a 6 vector

    """
    f = np.sqrt(2.0)
    A_voigt = np.array([1., 1., 1., f, f, f])*A_mandel
    if(order == 'voigt'):
        # Voigt in --> Reordering needed
        A_voigt = A_voigt[np.array((0, 1, 2, 5, 4, 3))]
    return A_voigt

def Mandel2VoigtStress( A_mandel, order = 'voigt' ):
    """Convert a tensor in Mandel notation to Voigt (for stresses).
    
    Parameters
    ----------
    A_voigt : ndarray
        symmetric 2-tensor in Mandel notation

    order : str
        if 'voigt' the order (xx, yy, zz, yz, xz, xy) and
        otherwise, (xx, yy, zz, xy, xz, yz)  is returned on output;

    Returns
    -------
    ndarray
        The converted tensor as a 6 vector

    """
    f = np.sqrt(0.5)
    A_voigt = np.array([1., 1., 1., f, f, f])*A_mandel
    if(order == 'voigt'):
        # Voigt in --> Reordering needed
        A_voigt[:] = A_voigt[np.array((0, 1, 2, 5, 4, 3))]
    return A_voigt

def StiffnessVoigt2Mandel( C_v, order = 'voigt' ):
    C_m = np.array(C_v)
    f = np.sqrt(2.0)
    C_m[:, :3] *= f
    C_m[:3, :] *= f
    if( order == 'voigt' ):
        idx = np.array((0, 1, 2, 5, 4, 3))
        C_m = C_m[idx[:, None], idx[None, :]]
    return C_m

def ComplianceVoigt2Mandel( S_v, order = 'voigt' ):
    S_m = np.array(S_v)
    f = np.sqrt(0.5)
    S_m[:, :3] *= f
    S_m[:3, :] *= f
    if( order == 'voigt' ):
        idx = np.array((0, 1, 2, 5, 4, 3))
        S_m = S_m[idx[:, None], idx[None, :]]
    return S_m

def Full2Mandel(A):
    f = np.sqrt(2.0)
    A_mandel = np.array([A[0,0], A[1,1], A[2,2], f*A[0,1], f*A[0,2], f*A[1,2]])
    return A_mandel

def Mandel2Full(A_mandel):
    f = np.sqrt(0.5)
    idx = np.array(((0, 3, 4), (3, 1, 5), (4, 5, 2)))
    A = np.zeros((3, 3))
    A = f*A_mandel[idx]
    # for i in range(3):
    #     for j in range(3):
    #         A[i, j] = f*A_mandel[idx[i][j]]
    f = np.sqrt(2.0)
    A[0, 0] *= f 
    A[1, 1] *= f 
    A[2, 2] *= f 
    return A

def IsoProjectionKappa(A_mandel):
    """Project 2-tensor in Mandel notation onto Id.
    
    The computation computes the orthogonal projection of an arbitrary, symmetric 2 tensor
    encoded as a 6-vector in Mandel notation onto the Id (the 2nd order identit ytensor).
    
    If given a 2d array, the first index is assumed to represent different microstructures
    and the projection is computed in vectorized form, returning a numpy.ndarray.

    Parameters
    ----------
    A_mandel : ndarray
        If ndim=1, then a single tensor is supplied in terms of a 6 vector according to the Mandel notation.
        If ndim=2, shape = (n, 6,), then n different conductivity tensors are provided in the same notation.

    Returns
    -------
    ndarray :
        If ndim=1, a scalar isotropic conductivity is returned.
        If ndim=2, a numpy.ndarray containing the n projections is returned.
    """
    if(A_mandel.ndim==1):
        kappa = A_mandel[:3].mean()
    else:
        # vectorized computation
        kappa = A_mandel[:,:3].mean(axis=1)
    return kappa

def IsoProjectionC(C_mandel):
    """Project 4-tensor in Mandel notation onto isotropic projectors.
    
    The computation computes the orthogonal projection of an arbitrary, symmetric 4-tensor
    encoded as a 6x6 matrix in Mandel notation onto the two isotropic projectors to compute
    the bulk modulus K and the shear modulus G.
    
    If given a 3d array, the first index is assumed to represent different microstructures
    and the projection is computed in vectorized form, returning a numpy.ndarray.

    Parameters
    ----------
    A_mandel : ndarray
        If ndim=2, then a single tensor is supplied in terms of a 6x6 matrix according to the Mandel notation.
        If ndim=3, shape = (n, 6, 6,), then n different bulk and shear moduli are returned.

    Returns
    -------
    float or ndarray :
        If ndim=2, the bulk modulus is returned.
        If ndim=3, a numpy.ndarray containing the n different bulk moduli is returned.
    float or ndarray :
        If ndim=2, the shear modulus is returned.
        If ndim=3, a numpy.ndarray containing the n different shear moduli is returned.
    """
    if(C_mandel.ndim==2):
        K = C_mandel[:3, :3].mean()
        G = (np.trace(C_mandel)-3.*K)/10.
    else:
        # vectorized computation
        K = C_mandel[:, :3, :3].mean(axis=(1, 2))
        G = (np.trace(C_mandel, axis1=1, axis2=2)-3.*K)/10.
    return K, G

def Piso1():
    """Returns the first isotropic projector in Mandel notation."""
    P = np.zeros((6, 6))
    P[:3,:3] = 1./3.
    return P

def Piso2():
    """Returns the second isotropic projector in Mandel notation."""
    P = np.eye(6)
    P = P - Piso1()
    return P

def Ciso(K, G):
    """Returns an isotropic stiffness tensor in Mandel notation."""
    if(type(K) is np.ndarray):
        return (3.*K-2.*G)[:, None, None]*Piso1()[None, :, :] + 2.*G[:, None, None]*np.eye(6)[None, :, :]
    return (3.*K-2.*G)*Piso1() + 2.*G*np.eye(6)

def test_Conversion():
    """Test various conversions, e.g., Full tensor <-> Mandel, Mandel <-> Voigt (both orderings)"""
    n_test = 10
    for i in range(n_test):
        # generate random tensor
        A = np.random.uniform(-1, 1, size=(3, 3))
        A = A+A.T
        A_m = Full2Mandel(A)
        assert(np.linalg.norm(A - Mandel2Full(A_m))<1.e-10), "Full->Mandel->Full failed for " + str(A)
        A_m = np.random.uniform(-1, 1, size=(6,))
        assert(np.linalg.norm(A_m - Full2Mandel(Mandel2Full(A_m)))<1.e-10), "Mandel->Full->Mandel failed for A_m=" + str(A_m)

        assert(np.linalg.norm(A_m - Voigt2MandelStrain(Mandel2VoigtStrain(A_m)))<1.e-10), "Mandel->Voigt_eps->Mandel failed for A_m=" + str(A_m)
        assert(np.linalg.norm(A_m - Voigt2MandelStress(Mandel2VoigtStress(A_m)))<1.e-10), "Mandel->Voigt_sig->Mandel failed for A_m=" + str(A_m)

        assert(np.linalg.norm(A_m - Voigt2MandelStrain(Mandel2VoigtStrain(A_m, order='abq'), order='abq'))<1.e-10), "Mandel->Voigt_eps,ABQ->Mandel failed for A_m=" + str(A_m)
        assert(np.linalg.norm(A_m - Voigt2MandelStress(Mandel2VoigtStress(A_m, order='abq'), order='abq'))<1.e-10), "Mandel->Voigt_sig,ABQ->Mandel failed for A_m=" + str(A_m)

def test_Ciso():
    """test the isotropic projection for 4-tensors using randomized data"""
    n_test = 10
    G = np.random.uniform(0, 10, size=n_test)
    K = np.random.uniform(0, 10, size=n_test)
    for k, g in zip(K, G):
        C = Ciso(k, g)
        k_fit, g_fit = IsoProjectionC(C)
        assert( np.abs(k-k_fit) + np.abs(g-g_fit) < 1.e-10 ), f"Error in isotropic projection (4-tensor): K, G true = [{k}, {g}]; K, G projected = [{k_fit}, {g_fit}]"

def test_kappaiso():
    """test the isotropic projection for 2-tensors using randomized data"""
    n_test = 10
    Id_m = np.array((1., 1., 1., 0., 0., 0.))
    for kappa in np.random.uniform(0, 10, size=n_test):
        kappa_fit = IsoProjectionKappa( kappa*Id_m )
        assert( np.abs(kappa-kappa_fit) < 1.e-10 ), f"Error in isotropic projection (2-tensor): kappa = {kappa}; kappa projected = {kappa_fit}"

def safeParam(name, d):
    """Safely extract data from a dictionary"""
    r = None
    if(name in d):
        r =d[name]
    return (r is not None), r

def ConvertElasticConstants(**kwargs):
    # todo: check for K, G, E, nu
    # if any two are available, compute other parameters (6 cases)
    el_const = {"E": None, "nu": None, "G": None, "K": None}
    el_const.update(kwargs)
    E, K, G, nu = el_const["E"], el_const["K"], el_const["G"], el_const["nu"]
    has_K = K is not None
    has_E = E is not None
    has_G = G is not None
    has_nu = nu is not None
    if(has_K+has_G+has_E+has_nu < 2):
        raise ValueError("Insufficient inputs: at least two independent elastic constants (E, K, G, nu) required, received: " + str(kwargs))
    if has_E:
        assert(E>0), f"Youngs modulus must be positive, but received E={E}"
    if has_K:
        assert(K>0), f"Bulk modulus must be positive, but received K={K}"
    if has_G:
        assert(G>0), f"Shear modulus must be positive, but received G={G}"
    if has_nu:
        assert((nu>-1.) and (nu<0.5)), f"Poisson ratio must satisfy -1 < nu < 0.5, but received nu={nu}"

    if(not has_E):
        if(has_K and has_G):
            E = 9.*K*G/(3.*K+G)
            nu = E/(2.*G) - 1.
        elif(has_K and has_nu):
            E = K*3.*(1.-2.*nu)
            G = E/(2.*(1.+nu))
        else:
            E = G*2.*(1.+nu)
            K = E/(3.*(1.-2.*nu))
    else:
        if(has_nu):
            # E, nu given
            G = E/(2.*(1.+nu))
            K = E/(3.*(1.-2.*nu))
        else:
            if(has_K):
                # E, K given
                nu = (3.*K - E)/(6.*K)
                G = E/(2.*(1.+nu))
            else:
                # E, G given
                nu = E/(2.*G) - 1.
                K = E/(3.*(1.-2.*nu))
    el_const["K"]=K
    el_const["E"]=E
    el_const["G"]=G
    el_const["nu"]=nu

    return el_const