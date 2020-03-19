from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import pandas as pd
import os
import analysis_functions as af



data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'

zt, rho_c = af.transformed_coords(data_path)
t,z,x=rho_c.shape

def transform_and_interp(k):
    data = rho_c[k]
    dummy = np.zeros((z,x)) #variable to store date during transform
    
    #applying the transform on the data
    for i in range(x):
            for j in range(z):
                wanted_data=data[j,i]
                z_loc = int(zt[j,i])
                if z_loc<580:
                    dummy[z_loc,i]=wanted_data
    
    dummy[dummy==0]=np.nan
    
    #now interpolating data
    dummy_frame = pd.DataFrame(dummy)
    dummy_int = dummy_frame.interpolate()
    return dummy_int.values


if __name__ == '__main__':
      numbers=range(t)
      p = Pool()
      result = p.map(transform_and_interp, numbers)
      p.close()
      p.join()

result=np.dstack(result)

np.savez('{}/transformation_result'.format(os.path.dirname(data_path)),result=result)
