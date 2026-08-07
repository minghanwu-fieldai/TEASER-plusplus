## Pairwise Global Pose

We assume scale=1. Suppose scans A and B have TIM $\{(\bar{a}_k, \bar{b}_k)\}$ and raw points $\{(a_k,b_k)\}$. Suppose the rotation parts of the current global poses of them are $R_A,R_B$ respectively. We can rewrite the TLS for finding the global rotation of B as
$$\min_{R^T\in SO(3)}\sum_{k=1}^K \min(\frac{\|\bar{b}_k-R^T R_A \bar{a}_k\|^2}{\delta_k^2},1)$$

where $\delta_k$ is the noise upper bound (all equal in practice).

The only differences from the current formulation are
1) We optimize the pose of target instead of source
2) We need to express the source measurements in global frame first
3) We optimize over $R^T$ directly and then assign $\hat{R}_B = R$

Then, the component-wise translation estimation should be modified in the same way: 
$$\min_{t_j} \sum_{k=1}^K \min(\frac{(t_j-[b_k-\hat{R}_B^T(R_Aa_k+t_A)]_j)^2}{\delta_k^2},1)$$
and assign $\hat{t}_B=-\hat{R}_B t^*$.

Rotation and translation can be solved sequentially in the original way after this modification.

## Multi-edge Global Pose

Now suppose scan B has overlapping neighbors forming set $N$, each neighbor having its own $R_i,t_i$ and inlier correspondence set. We sum the terms up to form an aggregated TLS:
$$\min_{R^T\in SO(3)}\sum_{i\in N}\sum_{k=1}^{K_i} \min(\frac{\|\bar{b}^i_k-R^T R_i \bar{a}^i_k\|^2}{\delta_k^2},1)$$
$$\min_{t_j} \sum_{i\in N}\sum_{k=1}^{K_i} \min(\frac{(t_j-[b^i_k-\hat{R}_B^T(R_ia^i_k + t_i)]_j)^2}{\delta_k^2},1)$$

We iterate through the node set, setting all other node poses as constant when optimizing pose for node B.

$$\argmin_{\{R_i\}_{i\in V}}\sum_{(p,q)\in E}\sum_{j\in \text{Corr}(p,q)} w_j \|R_py_j-R_qx_j\|^2$$
where V is the set of point clouds, each $R_i$ is an SO(3) rotation, E is the cloud adjacency edges, $Corr(p,q)$ is the set of point correspondences in the overlap of clouds p and q, and $(x_j,y_j)$ are the two points in one correspondence.