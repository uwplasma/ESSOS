import os
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.surfaces import SurfaceRZFourier
from essos.qfm import QfmSurface 
from essos.fields import Vmec, BiotSavart

method = 'slsqp'  #slsqp lbfgs

# 1. 加载等离子体 VMEC 文件，并生成 surface
ntheta=30
nphi=30
vmec = os.path.join('input_files','input.rotating_ellipse')
surf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period')
initial_vol = surf.volume
# 2. 创建初始线圈并生成 BiotSavart 磁场


ntheta=35
nphi=35

# Initialize VMEC field
truesurf = SurfaceRZFourier(vmec, ntheta=ntheta, nphi=nphi, range_torus='half period')


truevmec = Vmec(os.path.join(os.path.dirname(__file__), 'input_files', 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'),
            ntheta=ntheta, nphi=nphi, range_torus='half period')


from essos.coils import Coils_from_json
coils = Coils_from_json("stellarator_coils.json")

# 创建磁场对象
field = BiotSavart(coils)

# 3. 计算当前体积作为 target label（或设置固定值）
target_volume = truevmec.surface.volume  # 你可以手动设置一个目标值，如 target_volume = 1.0

# 4. 构建 QfmSurface 优化器
qfm = QfmSurface(
    field=field,
    surface=surf,
    label='volume',         # or "area"
    targetlabel=target_volume  # or target_area
)

# 5. 运行优化（选择方法）
result = qfm.run(tol=1e-3, maxiter=10000,method=method)

# 6. 打印结果
print("Optimization method:", method)
print("Optimization success:", result['success'])
print("Final Bnormal objective:", result['fun'])
print("Iterations:", result['iter'])
print(f"target volume: {target_volume}, initial volume: {initial_vol}, final volume: {result['s'].volume}")




fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
coils.plot(ax=ax1, show=False)
truesurf.plot(ax=ax1, show=False)
coils.plot(ax=ax2, show=False)
result['s'].plot(ax=ax2, show=False)
plt.tight_layout()
plt.show()
