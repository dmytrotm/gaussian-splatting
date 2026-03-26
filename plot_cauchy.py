import json
import matplotlib.pyplot as plt

# Data
iters = [1000, 2000, 3000, 4000]

base_psnr = [21.31, 22.75, 23.78, 24.23]
base_lpips = [0.505, 0.415, 0.354, 0.316]

act_psnr = [19.44, 19.02, 18.47, 19.19]
act_lpips = [0.510, 0.436, 0.390, 0.346]

loss_psnr = [22.41, 23.86, 24.82, 25.83]
loss_lpips = [0.499, 0.371, 0.286, 0.224]

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(iters, base_psnr, label='Baseline (Entropy Only)', marker='o', color='#3498db')
ax1.plot(iters, act_psnr, label='Cauchy Act + Entropy', marker='s', color='#e74c3c')
ax1.plot(iters, loss_psnr, label='Cauchy Loss + Entropy', marker='^', color='#2ecc71')
ax1.set_title('PSNR Comparison (first 4000 iters)')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('PSNR (Higher is Better)')
ax1.legend()
ax1.grid(alpha=0.2)

ax2.plot(iters, base_lpips, label='Baseline (Entropy Only)', marker='o', color='#3498db')
ax2.plot(iters, act_lpips, label='Cauchy Act + Entropy', marker='s', color='#e74c3c')
ax2.plot(iters, loss_lpips, label='Cauchy Loss + Entropy', marker='^', color='#2ecc71')
ax2.set_title('LPIPS Sharpness (first 4000 iters)')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('LPIPS (Lower is Better)')
ax2.legend()
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('cauchy_comparison.png', dpi=150, bbox_inches='tight')
