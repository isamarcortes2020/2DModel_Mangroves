from landlab import RasterModelGrid, imshow_grid
from landlab.components import LinearDiffuser
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio




def dataImport(rasterLayer):
    File = rio.open(rasterLayer)
    vegetation_array = File.read(1)
    vegetation_array = vegetation_array.astype(np.int64)
    return vegetation_array




VegData = dataImport('/Users/isamarcortes/Dropbox/Isamar/Papers_In_Prep/Paper_4/RasterLayersForLandlab/PR1_Paper4.tif')


#########Constant for each island
sal_array = np.select([VegData == 0, VegData == 1], [35, 36], VegData)###make a salinty grid
Island = RasterModelGrid((VegData.shape))
vegetation = Island.add_field('vegetation',VegData,at='node')
qs = Island.add_zeros("salinity_flux", at="link")
salinity = Island.add_zeros('salinity', at='node')
Island.status_at_node[VegData.flatten()==0] = Island.BC_NODE_IS_FIXED_VALUE

#salinity = Island.add_field('salinity',sal_array,at='node')
#Island.status_at_node[salinity == 35]=Island.BC_NODE_IS_FIXED_VALUE
#Island.status_at_node[salinity==36]=Island.BC_NODE_IS_CORE

def outerEdgeSalinity(salStart=None, salEnd=None, salinity=None):
    # 1. Check if everything is missing
    if salStart is None and salEnd is None and salinity is None:
        return 40 
    
    # 2. If we reach this point, we assume the user provided data.
    # We should still provide internal defaults for salStart/End 
    # just in case only 'salinity' was provided.
    s_start = salStart if salStart is not None else 30
    s_end = salEnd if salEnd is not None else 35
    
    # 3. Handle the salinity array
    if salinity is None:
        # If they gave start/end but no array, what size should it be?
        # Defaulting to a size of 1 for now.
        return np.array([s_start]) 
    
    # 4. Calculate based on the provided array
    salinitySize = salinity.size
    return np.linspace(s_start, s_end, salinitySize)

#sal_values = 40
sal_values = outerEdgeSalinity(35,40,salinity)###change based on whether inner or outer bay island
salinity[:]=sal_values
Enet = 1.2 #constant for all islands
D = Island.add_zeros("D", at="node")

def difussionOuterBay(slope,xLengthArray,D1,D2):
    xLengthArray = xLengthArray.shape[0]
    D[Island.y_of_node > slope * Island.x_of_node + xLengthArray] = D1
    D[Island.y_of_node <= slope * Island.x_of_node + xLengthArray] = D2
    return D

D =difussionOuterBay(-1,sal_array,34,10)
f = D.reshape(sal_array.shape)
plt.imshow(f)
plt.colorbar()


D_at_link = Island.map_mean_of_link_nodes_to_link(D)


max_D = np.max(D)
dx = Island.dx
dt_stable = 0.2 * (dx**2) / max_D  # 0.2 is a safety factor
print(f"Stable dt is: {dt_stable}")
#dt = 1 * Island.dx * Island.dx / D
#gradients = Island.calc_grad_at_link(salinity)
#qs[Island.active_links] = -D * gradients[Island.active_links]
#dy = -Island.calc_flux_div_at_node(qs)
#salinity[Island.core_nodes] = (salinity[Island.core_nodes] +Enet)




for i in range(100000):
    g = Island.calc_grad_at_link(salinity)
    #D1 = Island.calc_grad_at_link(D)
    qs[Island.active_links] = -D_at_link[Island.active_links] * g[Island.active_links]
    dqdx = Island.calc_flux_div_at_node(qs)
    dsdt = -dqdx + Enet
    salinity[Island.core_nodes] = salinity[Island.core_nodes] + (dsdt[Island.core_nodes]*dt_stable)

salinityBinary = salinity
salinityBinary[salinityBinary>100]=0
salinityBinary[salinityBinary<41]=0
#salinityBinary[salinityBinary==85.000011]=0



t = salinityBinary.reshape(sal_array.shape) ###change this to size of salinity
count = np.count_nonzero(t)
Island.imshow(t)

