import numpy as np

### COM centered unit cells ###
uc_0_centered = np.empty((8,3))
uc_51_centered = np.empty((8,3))

uc_0_centered[0,0] = 2.7612519805334728
uc_0_centered[0,1] = -1.9525000000000003
uc_0_centered[0,2] = -0.013810814424747075

uc_0_centered[1,0] = 2.7612519805334728
uc_0_centered[1,1] = 1.9525000000000003
uc_0_centered[1,2] = -0.013810814424747075

uc_0_centered[2,0] = 0
uc_0_centered[2,1] = -1.9525000000000003
uc_0_centered[2,2] = 2.7474411661087172

uc_0_centered[3,0] = 0
uc_0_centered[3,1] = 1.9525000000000003
uc_0_centered[3,2] = 2.7474411661087172

uc_0_centered[4,0] = -2.7612519805334728
uc_0_centered[4,1] = -1.9525000000000003
uc_0_centered[4,2] = -0.013810814424747075

uc_0_centered[5,0] = -2.7612519805334728
uc_0_centered[5,1] = 1.9525000000000003
uc_0_centered[5,2] = -0.013810814424747075

uc_0_centered[6,0] = 0
uc_0_centered[6,1] = -1.9525000000000003
uc_0_centered[6,2] = -2.7198195372592227

uc_0_centered[7,0] = 0
uc_0_centered[7,1] = 1.9525000000000003
uc_0_centered[7,2] = -2.7198195372592227

uc_51_centered[0,0] = 2.7612519805334728
uc_51_centered[0,1] = -1.9525000000000003
uc_51_centered[0,2] = 1.7763568394002505E-15
       
uc_51_centered[1,0] = 2.7612519805334728
uc_51_centered[1,1] = 1.9525000000000003
uc_51_centered[1,2] = 1.7763568394002505E-15
       
uc_51_centered[2,0] = 0
uc_51_centered[2,1] = -1.9525000000000003
uc_51_centered[2,2] = 2.7612519805334674
       
uc_51_centered[3,0] = 0
uc_51_centered[3,1] = 1.9525000000000003
uc_51_centered[3,2] = 2.7612519805334674
       
uc_51_centered[4,0] = -2.7612519805334728
uc_51_centered[4,1] = -1.9525000000000003
uc_51_centered[4,2] = 1.7763568394002505E-15
       
uc_51_centered[5,0] = -2.7612519805334728
uc_51_centered[5,1] = 1.9525000000000003
uc_51_centered[5,2] = 1.7763568394002505E-15
       
uc_51_centered[6,0] = 0
uc_51_centered[6,1] = -1.9525000000000003
uc_51_centered[6,2] = -2.761251980533471

uc_51_centered[7,0] = 0
uc_51_centered[7,1] = 1.9525000000000003
uc_51_centered[7,2] = -2.761251980533471


diff = uc_0_centered - uc_51_centered

diff_loss = 0
for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        diff_loss += diff[i,j]**2

print(diff)
print(diff_loss)


### uncentered unit cells ###
a = 3.905

uc_0_uncentered = np.empty((8,3))
uc_51_uncentered = np.empty((8,3))

uc_0_uncentered[0,0] = 138.06259902667335
uc_0_uncentered[0,1] = 0
uc_0_uncentered[0,2] = 2.706008722834476

uc_0_uncentered[1,0] = 138.06259902667335
uc_0_uncentered[1,1] = 3.9050000000000007
uc_0_uncentered[1,2] = 2.706008722834476

uc_0_uncentered[2,0] = 135.30134704613988
uc_0_uncentered[2,1] = 0
uc_0_uncentered[2,2] = 5.4672607033679403

uc_0_uncentered[3,0] = 135.30134704613988
uc_0_uncentered[3,1] = 3.9050000000000007
uc_0_uncentered[3,2] = 5.4672607033679403

uc_0_uncentered[4,0] = 132.54009506560641
uc_0_uncentered[4,1] = 0
uc_0_uncentered[4,2] = 2.706008722834476

uc_0_uncentered[5,0] = 132.54009506560641
uc_0_uncentered[5,1] = 3.9050000000000007
uc_0_uncentered[5,2] = 2.706008722834476

uc_0_uncentered[6,0] = 135.30134704613988
uc_0_uncentered[6,1] = 0
uc_0_uncentered[6,2] = 6.1005448828337803E-16

uc_0_uncentered[7,0] = 135.30134704613988
uc_0_uncentered[7,1] = 3.9050000000000007
uc_0_uncentered[7,2] = 6.1005448828337803E-16

uc_51_uncentered[0,0]= 138.06259902667335
uc_51_uncentered[0,1]= 0
uc_51_uncentered[0,2]= 8.228512683901414
       
uc_51_uncentered[1,0]= 138.06259902667335
uc_51_uncentered[1,1]= 3.9050000000000007
uc_51_uncentered[1,2]= 8.228512683901414
       
uc_51_uncentered[2,0]= 135.30134704613988
uc_51_uncentered[2,1]= 0
uc_51_uncentered[2,2]= 10.98976466443488
       
uc_51_uncentered[3,0]= 135.30134704613988
uc_51_uncentered[3,1]= 3.9050000000000007
uc_51_uncentered[3,2]= 10.98976466443488
       
uc_51_uncentered[4,0]= 132.54009506560641
uc_51_uncentered[4,1]= 0
uc_51_uncentered[4,2]= 8.228512683901414
       
uc_51_uncentered[5,0]= 132.54009506560641
uc_51_uncentered[5,1]= 3.9050000000000007
uc_51_uncentered[5,2]= 8.228512683901414
       
uc_51_uncentered[6,0]= 135.30134704613988
uc_51_uncentered[6,1]= 0
uc_51_uncentered[6,2]= 5.4672607033679412

uc_51_uncentered[7,0]= 135.30134704613988
uc_51_uncentered[7,1]= 3.9050000000000007
uc_51_uncentered[7,2]= 5.4672607033679412
