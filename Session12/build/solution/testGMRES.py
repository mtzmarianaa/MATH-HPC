from GMRES_MGS import *
from numpy.linalg import eig
import matplotlib.pyplot as plt

plt.ion()

# For the Vico Greengard paper:
# Note that this takes a while to run
# VG_mat = np.genfromtxt("VG_mat.csv", dtype=complex)
# VG_mat = np.real(VG_mat)
# VG_b = np.genfromtxt("VG_b.csv", dtype=complex)
# VG_b = np.real(VG_b)

# For the close-to-touching interactions paper
CTT_mat = np.genfromtxt("CTT_mat.txt", delimiter = ',')
CTT_b = np.genfromtxt("CTT_b.csv", delimiter = ',')

# For the MNIST data set
RBF_b = np.genfromtxt("RBF_b.csv", delimiter = ',')
RBF0_mat = np.genfromtxt("RBF0_mat.csv", delimiter = ',')
RBF1_mat = np.genfromtxt("RBF1_mat.csv", delimiter = ',')
RBF2_mat = np.genfromtxt("RBF2_mat.csv", delimiter = ',')

# Get eigenvalues
maxgap = 4*[0]
# eig_VG = eig(VG_mat)[0]
# maxgap[0] = max([eig_VG[i]/eig_VG[i+1] for i in range(len(eig_VG)-1) ] )
eig_CTT = eig(CTT_mat)[0]
maxgap[0] = max([eig_CTT[i]/eig_CTT[i+1] for i in range(len(eig_CTT)-1) ] )
eig_RBF0 = eig(RBF0_mat)[0]
maxgap[1] = max([eig_RBF0[i]/eig_RBF0[i+1] for i in range(len(eig_RBF0)-1) ] )
eig_RBF1 = eig(RBF1_mat)[0]
maxgap[2] = max([eig_RBF1[i]/eig_RBF1[i+1] for i in range(len(eig_RBF1)-1) ] )
eig_RBF2 = eig(RBF2_mat)[0]
maxgap[3] = max([eig_RBF2[i]/eig_RBF2[i+1] for i in range(len(eig_RBF2)-1) ] )

# Use GMRES
ms = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
tol = 1e-10
l = len(ms)
#iterVG = 10*[0]
iterCTT = l*[0]
iterRBF0 = l*[0]
iterRBF1 = l*[0]
iterRBF2 = l*[0]
#errVG = l*[0]
errCTT = l*[0]
errRBF0 = l*[0]
errRBF1 = l*[0]
errRBF2 = l*[0]


for i in range(l):
    #VG_x, j, err = gmres_mgs(VG_mat, VG_b, VG_b, ms[i], tol)
    #iterVG[i] = j
    #errVG[i] = err
    CTT_x, j, err = gmres_mgs(CTT_mat, CTT_b, CTT_b, ms[i], tol)
    iterCTT[i] = j
    errCTT[i] = err
    RBF0_x, j, err = gmres_mgs(RBF0_mat, RBF_b, RBF_b, ms[i], tol)
    iterRBF0[i] = j
    errRBF0[i] = err
    RBF1_x, j, err = gmres_mgs(RBF1_mat, RBF_b, RBF_b, ms[i], tol)
    iterRBF1[i] = j
    errRBF1[i] = err
    RBF2_x, j, err = gmres_mgs(RBF2_mat, RBF_b, RBF_b, ms[i], tol)
    iterRBF2[i] = j
    errRBF2[i] = err

# Plot
plt.figure(figsize=(8, 6), dpi=80)
plt.loglog(iterCTT, errCTT, c = "#00d4e9", marker = 'o', label = 'Close-to-touching')
plt.loglog(iterRBF0, errRBF0, c = "#004dff", marker = 'o', label = 'RBF0')
plt.loglog(iterRBF1, errRBF0, c = "#0a00ad", marker = 'o', label = 'RBF1')
plt.loglog(iterRBF2, errRBF0, c = "#242b4f", marker = 'o', label = 'RBF2')
plt.legend()
plt.title("Number of GMRES iterations and L2 error")
plt.xlabel("GMRES iterations")
plt.ylabel("L2 error")


errsLast = [errCTT[-1], errRBF0[-1], errRBF1[-1], errRBF2[-1]]
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(maxgap, errsLast, c = "#7100e9", marker = 'o')
plt.title("Maximum spectral gap and L2 error")
plt.xlabel(r'$\max_{i} \frac{\lambda_{i}}{\lambda_{i+1}}$')
plt.ylabel("L2 error")

