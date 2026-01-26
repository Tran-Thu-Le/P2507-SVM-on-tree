// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <stdexcept>

// namespace py = pybind11;

// //====================== CSR (undirected tree) ======================//
// struct CSR {
//     int n;
//     std::vector<long long> indptr;
//     std::vector<int> indices;
//     std::vector<double> weights;
// };

// static CSR build_csr(int n, const std::vector<std::tuple<int,int,double>>& edges) {
//     std::vector<int> deg(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;
//         deg[u]++; deg[v]++;
//     }
//     CSR csr; csr.n = n;
//     csr.indptr.assign(n + 1, 0);
//     for (int i=0;i<n;i++) csr.indptr[i+1] = csr.indptr[i] + deg[i];
//     csr.indices.assign(csr.indptr.back(), 0);
//     csr.weights.assign(csr.indptr.back(), 0.0);

//     std::vector<int> fill(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;
//         long long pu = csr.indptr[u] + fill[u];
//         csr.indices[pu] = v; csr.weights[pu] = w; fill[u]++;
//         long long pv = csr.indptr[v] + fill[v];
//         csr.indices[pv] = u; csr.weights[pv] = w; fill[v]++;
//     }
//     return csr;
// }

// //====================== Rooting & orders ======================//
// static void root_tree_iter(
//     const CSR& G, int root,
//     std::vector<int>& parent, std::vector<double>& parent_w,
//     std::vector<int>& depth, std::vector<int>& order_post
// ){
//     const int n = G.n;
//     parent.assign(n, -1);
//     parent_w.assign(n, 0.0);
//     depth.assign(n, 0);
//     std::vector<int> stack; stack.reserve(n);
//     std::vector<int> order_pre; order_pre.reserve(n);

//     stack.push_back(root);
//     parent[root] = -1;
//     while (!stack.empty()) {
//         int u = stack.back(); stack.pop_back();
//         order_pre.push_back(u);
//         for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
//             int v = G.indices[k];
//             if (v == parent[u]) continue;
//             parent[v] = u;
//             parent_w[v] = G.weights[k];
//             depth[v] = depth[u] + 1;
//             stack.push_back(v);
//         }
//     }
//     order_post = order_pre;
//     std::sort(order_post.begin(), order_post.end(),
//               [&](int a, int b){ return depth[a] > depth[b]; });
// }

// static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
//     std::vector<uint8_t> contains(parent.size(), 0);
//     for (int u = target; u != -1; u = parent[u]) contains[u] = 1;
//     return contains;
// }

// //====================== sz & noise accum on root->target path ======================//
// static void accumulate_noise_on_branch(
//     const CSR& G,
//     int root, int target,
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     const std::vector<int>& colors,
//     std::vector<double>& dis0,
//     std::vector<double>& dis1,
//     std::vector<double>& maxtr, // contribution of the child-on-path subtree at u
//     double& total_noise_root    // tol at root (class-consistent)
// ){
//     const int n = G.n;
//     std::vector<uint8_t> on_path = mark_path_to_target(parent, target);
//     dis0.assign(n, 0.0); dis1.assign(n, 0.0);
//     maxtr.assign(n, 0.0);

//     // compute subtree sizes (counts) by a bottom-up pass
//     std::vector<long long> sz0(n,0), sz1(n,0);
//     for (int u : order_post) {
//         if (colors[u] == 0) sz0[u] += 1; else sz1[u] += 1;
//     }
//     for (int u : order_post) {
//         int p = parent[u];
//         if (p != -1){ sz0[p] += sz0[u]; sz1[p] += sz1[u]; }
//     }

//     for (int u : order_post) {
//         for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
//             int v = G.indices[k];
//             if (parent[v] != u) continue;    // child
//             if (!on_path[v]) continue;       // only child on root->target path contributes
//             double w = G.weights[k];
//             dis0[u] += dis0[v] + (double)sz0[v] * w;
//             dis1[u] += dis1[v] + (double)sz1[v] * w;

//             if (colors[u] == colors[root]) {
//                 if (colors[root] == 0) maxtr[u] = dis0[v] + (double)sz0[v]*w;
//                 else                   maxtr[u] = dis1[v] + (double)sz1[v]*w;
//             }
//         }
//     }
//     total_noise_root = (colors[root]==0) ? dis0[root] : dis1[root];
// }

// //====================== Main: arbitrary lambda ======================//
// py::dict fit_core_lambda_any(
//     py::array_t<double, py::array::c_style | py::array::forcecast> X,
//     py::array_t<long long, py::array::c_style | py::array::forcecast> y,
//     double lambda_
// ){
//     auto bx = X.request(); auto by = y.request();
//     if (bx.ndim != 2) throw std::runtime_error("X must be 2D");
//     if (by.ndim != 1) throw std::runtime_error("y must be 1D");
//     const int64_t N = bx.shape[0]; const int64_t D = bx.shape[1];
//     if (by.shape[0] != N) throw std::runtime_error("X,y length mismatch");
//     if (N < 2) throw std::runtime_error("Need at least 2 samples");
//     const double* Xptr = static_cast<double*>(bx.ptr);
//     const long long* Yptr = static_cast<long long*>(by.ptr);

//     // means m0/m1, direction w
//     std::vector<double> m0(D,0.0), m1(D,0.0);
//     long long c0=0,c1=0;
//     for (int64_t i=0;i<N;i++){
//         if (Yptr[i]==0){ c0++; for (int64_t d=0; d<D; ++d) m0[d]+=Xptr[i*D+d]; }
//         else            { c1++; for (int64_t d=0; d<D; ++d) m1[d]+=Xptr[i*D+d]; }
//     }
//     if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 required.");
//     for (int64_t d=0; d<D; ++d){ m0[d]/=(double)c0; m1[d]/=(double)c1; }
//     std::vector<double> w(D,0.0);
//     for (int64_t d=0; d<D; ++d) w[d]=m1[d]-m0[d];
//     double nw=0.0; for(double v:w) nw+=v*v; nw=std::sqrt(nw);
//     if (nw==0.0){ w[0]=1.0; for (int64_t d=1; d<D; ++d) w[d]=0.0; } else
//         for (int64_t d=0; d<D; ++d) w[d]/=nw;

//     // projections and spoke lengths
//     std::vector<double> t(N,0.0), Lspoke(N,0.0);
//     std::vector<double> Xproj(N*D,0.0);
//     for (int64_t i=0;i<N;i++){
//         double ti=0.0;
//         for (int64_t d=0; d<D; ++d) ti += (Xptr[i*D+d]-m0[d])*w[d];
//         t[i]=ti;
//         for (int64_t d=0; d<D; ++d) Xproj[i*D+d]=m0[d]+ti*w[d];
//         double res2=0.0;
//         for (int64_t d=0; d<D; ++d){
//             double r = Xptr[i*D+d]-Xproj[i*D+d];
//             res2 += r*r;
//         }
//         Lspoke[i]=std::sqrt(res2);
//     }

//     // sort by t (spine order)
//     std::vector<int> idx(N); for(int i=0;i<N;i++) idx[i]=i;
//     std::sort(idx.begin(), idx.end(), [&](int a,int b){ return t[a]<t[b]; });

//     // spine cumulative coordinate S (0-based over N projections)
//     std::vector<double> S(N,0.0);
//     for (int k=1;k<N;k++){
//         S[k]=S[k-1]+std::abs(t[idx[k]]-t[idx[k-1]]);
//     }

//     // Build tree nodes: [0..N-1]=original, [N..2N-1]=projection
//     const int TOT = (int)(2*N);
//     std::vector<int> colors(TOT,0);
//     for (int i=0;i<N;i++){ colors[i]=(int)Yptr[i]; colors[N+i]=(int)Yptr[i]; }

//     // edges: spoke + spine
//     std::vector<std::tuple<int,int,double>> edges; edges.reserve(3*N);
//     for (int i=0;i<N;i++) edges.emplace_back(i, N+i, Lspoke[i]);
//     for (int k=0;k<N-1;k++){
//         int a = idx[k], b = idx[k+1];
//         edges.emplace_back(N+a, N+b, std::abs(t[b]-t[a]));
//     }
//     CSR G = build_csr(TOT, edges);

//     // mapping: spine order -> node id of projection
//     std::vector<int> spine_ids(N);
//     for (int k=0;k<N;k++) spine_ids[k]=N+idx[k];

//     //--- DP to get noise arrays along spine for each class and direction ---//
//     auto compute_side = [&](int root_node, int target_node, std::vector<double>& out_maxtr, double& tol_root){
//         std::vector<int> parent, depth, order_post;
//         std::vector<double> parent_w;
//         root_tree_iter(G, root_node, parent, parent_w, depth, order_post);

//         std::vector<double> d0,d1,maxtr;
//         double total_noise_root = 0.0;
//         accumulate_noise_on_branch(G, root_node, target_node, order_post, parent, colors,
//                                    d0,d1,maxtr,total_noise_root);
//         out_maxtr = maxtr;
//         tol_root = total_noise_root;
//     };

//     // tách spine theo lớp
//     std::vector<int> X0_sorted, X1_sorted; X0_sorted.reserve(N); X1_sorted.reserve(N);
//     for (int k=0;k<N;k++){
//         int pid = spine_ids[k];
//         if (colors[pid]==0) X0_sorted.push_back(pid); else X1_sorted.push_back(pid);
//     }

//     std::vector<double> max0L(TOT,0.0), max0R(TOT,0.0), max1L(TOT,0.0), max1R(TOT,0.0);
//     double dumpTol=0.0;
//     if (!X0_sorted.empty()){
//         int L = X0_sorted.front(), R = X0_sorted.back();
//         compute_side(L, R, max0L, dumpTol); // “đi về phải”
//         compute_side(R, L, max0R, dumpTol); // “đi về trái”
//     }
//     if (!X1_sorted.empty()){
//         int L = X1_sorted.front(), R = X1_sorted.back();
//         compute_side(L, R, max1L, dumpTol);
//         compute_side(R, L, max1R, dumpTol);
//     }

//     // vị trí projection trên spine: pos[node] in [0..N-1]
//     std::vector<int> pos(TOT, -1);
//     for (int k=0;k<N;k++){
//         int pid = spine_ids[k];
//         pos[pid]=k;
//     }
//     for (int i=0;i<N;i++){
//         pos[i] = pos[N+i]; // original chia sẻ pos với projection
//     }

//     // prefix-count số projections mỗi lớp trên spine
//     std::vector<int> cls_at(N,0);
//     for (int k=0;k<N;k++){
//         int pid = spine_ids[k]; cls_at[k]=colors[pid];
//     }
//     std::vector<int> pref0(N+1,0), pref1(N+1,0);
//     for (int k=0;k<N;k++){
//         pref0[k+1]=pref0[k]+(cls_at[k]==0);
//         pref1[k+1]=pref1[k]+(cls_at[k]==1);
//     }
//     int total0=pref0[N], total1=pref1[N];

//     auto left_same = [&](int c, int k)->int{
//         return (c==0)? pref0[k] : pref1[k];
//     };
//     auto right_same = [&](int c, int k)->int{
//         return (c==0)? (total0 - pref0[k+1]) : (total1 - pref1[k+1]);
//     };

//     auto noise_proj = [&](int cls, bool dir_right, int u_proj)->double{
//         if (cls==0) return dir_right ? max0L[u_proj] : max0R[u_proj];
//         else        return dir_right ? max1L[u_proj] : max1R[u_proj];
//     };

//     // f_side(u,v): nhiễu phía cùng nhãn với u khi root=u và đi theo nhánh tới v
//     auto f_side = [&](int u, int v)->double{
//         int cu = colors[u];
//         int iu = pos[u], iv = pos[v];
//         bool dir_right = (iv >= iu);
//         int u_proj = (u < (int)N) ? (N+u) : u;
//         double f = noise_proj(cu, dir_right, u_proj);

//         // Nếu u là original node, các lá cùng nhãn phía xét (kể cả u) còn đi thêm spoke(u).
//         // Ở bản này, spoke được cộng vào d(u,v) để tránh đếm đôi trong f.
//         (void)left_same; (void)right_same; // giữ cho compilers không cảnh báo nếu không dùng
//         return f;
//     };

//     // khoảng cách spine(u', v') + spoke(u) + spoke(v)
//     auto dist_uv = [&](int u, int v)->double{
//         int iu = pos[u], iv = pos[v];
//         double spine = std::abs(S[iv] - S[iu]);
//         double su = (u < (int)N) ? Lspoke[u] : 0.0;
//         double sv = (v < (int)N) ? Lspoke[v] : 0.0;
//         return spine + su + sv;
//     };

//     double best = std::numeric_limits<double>::infinity();
//     int best_u=-1, best_v=-1; double best_d=0.0;

//     for (int u=0; u<(int)TOT; ++u){
//         int cu = colors[u]; if (cu!=0 && cu!=1) continue;
//         for (int v=0; v<(int)TOT; ++v){
//             if (u==v) continue;
//             int cv = colors[v]; if (cv!=0 && cv!=1) continue;
//             if (cu == cv) continue;

//             double f = f_side(u,v) + f_side(v,u);
//             double d = dist_uv(u,v);
//             double L = f - lambda_ * d;
//             if (L < best){ best=L; best_u=u; best_v=v; best_d=d; }
//         }
//     }

//     auto to_orig = [&](int node)->int{ return (node < (int)N)? node : (node - (int)N); };
//     bool s_is_proj = (best_u >= (int)N);
//     bool p_is_proj = (best_v >= (int)N);

//     py::dict out;
//     out["s_node"]   = best_u;
//     out["p_node"]   = best_v;
//     out["s_is_proj"]= s_is_proj;
//     out["p_is_proj"]= p_is_proj;
//     out["s_orig"]   = to_orig(best_u);
//     out["p_orig"]   = to_orig(best_v);
//     out["min_val"]  = best;
//     out["dist"]     = best_d;
//     out["lambda"]   = lambda_;
//     return out;
// }

// PYBIND11_MODULE(svm_on_tree_cpp, m){
//     m.doc() = "C++ core for SVM-on-Tree (arbitrary lambda) — all DP/DFS inside C++";
//     m.def("fit_core_lambda_any", &fit_core_lambda_any,
//           py::arg("X"), py::arg("y"), py::arg("lamda")=1.0,
//           R"pbdoc(
//             Compute optimal (s,p) for arbitrary lambda.
//             Inputs:
//               X: (N,d) float64, y: (N,) int64 in {0,1}, lamda: float
//             Returns dict:
//               {s_node, p_node, s_is_proj, p_is_proj, s_orig, p_orig, min_val, dist, lambda}
//           )pbdoc");
// }

// svm_on_tree_cpp.cpp
// C++ core for SVM-on-Tree (arbitrary lambda), counting ONLY DFS+DP time.
// - Logic unchanged.
// - Optimizations: true postorder (no sort), single-pass accumulate, scan pairs only across opposite classes,
//   avoid building full Xproj, micro-alloc reserves.
// - Returned dict includes "t_dpdfs" (seconds) = sum of 4 calls to compute_side(...) only.

// svm_on_tree_cpp.cpp
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include <algorithm>
// #include <vector>
// #include <tuple>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <chrono>

// namespace py = pybind11;
// using hrc = std::chrono::high_resolution_clock;
// using secd = std::chrono::duration<double>;

// //====================== CSR (undirected tree) ======================//
// struct CSR {
//     int n{0};
//     std::vector<long long> indptr;
//     std::vector<int>       indices;
//     std::vector<double>    weights;
// };

// static CSR build_csr(int n, const std::vector<std::tuple<int,int,double>>& edges) {
//     std::vector<int> deg(n, 0);
//     for (auto &e : edges) {
//         int u,v; double w;
//         std::tie(u,v,w)=e; (void)w;
//         deg[u]++; deg[v]++;
//     }
//     CSR csr; csr.n = n;
//     csr.indptr.assign(n+1, 0);
//     for (int i=0;i<n;i++) csr.indptr[i+1] = csr.indptr[i] + deg[i];
//     csr.indices.assign(csr.indptr.back(), 0);
//     csr.weights.assign(csr.indptr.back(), 0.0);

//     std::vector<int> fill(n, 0);
//     for (auto &e : edges) {
//         int u,v; double w; std::tie(u,v,w)=e;
//         long long pu = csr.indptr[u] + fill[u];
//         csr.indices[pu]=v; csr.weights[pu]=w; fill[u]++;
//         long long pv = csr.indptr[v] + fill[v];
//         csr.indices[pv]=u; csr.weights[pv]=w; fill[v]++;
//     }
//     return csr;
// }

// //====================== Rooting & orders ======================//
// static void root_tree_iter(
//     const CSR& G, int root,
//     std::vector<int>& parent, std::vector<double>& parent_w,
//     std::vector<int>& depth, std::vector<int>& order_post
// ){
//     const int n = G.n;
//     parent.assign(n, -1);
//     parent_w.assign(n, 0.0);
//     depth.assign(n, 0);
//     std::vector<int> stack; stack.reserve(n);
//     std::vector<int> order_pre; order_pre.reserve(n);

//     stack.push_back(root);
//     parent[root] = -1;
//     while (!stack.empty()) {
//         int u = stack.back(); stack.pop_back();
//         order_pre.push_back(u);
//         for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
//             int v = G.indices[k];
//             if (v == parent[u]) continue;
//             parent[v] = u;
//             parent_w[v] = G.weights[k];
//             depth[v] = depth[u] + 1;
//             stack.push_back(v);
//         }
//     }
//     order_post = order_pre;
//     std::sort(order_post.begin(), order_post.end(),
//               [&](int a, int b){ return depth[a] > depth[b]; });
// }

// static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
//     std::vector<uint8_t> contains(parent.size(), 0);
//     for (int u = target; u != -1; u = parent[u]) contains[u] = 1;
//     return contains;
// }

// //====================== sz & noise accum on root->target path ======================//
// static void accumulate_noise_on_branch(
//     const CSR& G,
//     int root, int target,
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     const std::vector<int>& colors,
//     std::vector<double>& dis0,
//     std::vector<double>& dis1,
//     std::vector<double>& maxtr, // contribution of the child-on-path subtree at u
//     double& total_noise_root    // tol at root (class-consistent)
// ){
//     const int n = G.n;
//     std::vector<uint8_t> on_path = mark_path_to_target(parent, target);
//     dis0.assign(n, 0.0); dis1.assign(n, 0.0);
//     maxtr.assign(n, 0.0);

//     // subtree counts
//     std::vector<long long> sz0(n,0), sz1(n,0);
//     for (int u : order_post) {
//         if (colors[u] == 0) sz0[u] += 1; else sz1[u] += 1;
//     }
//     for (int u : order_post) {
//         int p = parent[u];
//         if (p != -1){ sz0[p] += sz0[u]; sz1[p] += sz1[u]; }
//     }

//     for (int u : order_post) {
//         for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
//             int v = G.indices[k];
//             if (parent[v] != u) continue;      // child
//             if (!on_path[v]) continue;         // chỉ nhánh nằm trên đường root->target
//             double w = G.weights[k];
//             dis0[u] += dis0[v] + (double)sz0[v] * w;
//             dis1[u] += dis1[v] + (double)sz1[v] * w;

//             if (colors[u] == colors[root]) {
//                 if (colors[root] == 0) maxtr[u] = dis0[v] + (double)sz0[v]*w;
//                 else                   maxtr[u] = dis1[v] + (double)sz1[v]*w;
//             }
//         }
//     }
//     total_noise_root = (colors[root]==0) ? dis0[root] : dis1[root];
// }

// //====================== Main: arbitrary lambda ======================//
// py::dict fit_core_lambda_any(
//     py::array_t<double, py::array::c_style | py::array::forcecast> X,
//     py::array_t<long long, py::array::c_style | py::array::forcecast> y,
//     double lambda_
// ){
//     auto bx = X.request(); auto by = y.request();
//     if (bx.ndim != 2) throw std::runtime_error("X must be 2D");
//     if (by.ndim != 1) throw std::runtime_error("y must be 1D");
//     const int64_t N = bx.shape[0]; const int64_t D = bx.shape[1];
//     if (by.shape[0] != N) throw std::runtime_error("X,y length mismatch");
//     if (N < 2) throw std::runtime_error("Need at least 2 samples");
//     const double* Xptr = static_cast<double*>(bx.ptr);
//     const long long* Yptr = static_cast<long long*>(by.ptr);

//     // ====== (1) means & spine direction ======  [không tính thời gian]
//     std::vector<double> m0(D,0.0), m1(D,0.0);
//     long long c0=0,c1=0;
//     for (int64_t i=0;i<N;i++){
//         if (Yptr[i]==0){ c0++; for (int64_t d=0; d<D; ++d) m0[d]+=Xptr[i*D+d]; }
//         else            { c1++; for (int64_t d=0; d<D; ++d) m1[d]+=Xptr[i*D+d]; }
//     }
//     if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 required.");
//     for (int64_t d=0; d<D; ++d){ m0[d]/=(double)c0; m1[d]/=(double)c1; }
//     std::vector<double> w(D,0.0);
//     for (int64_t d=0; d<D; ++d) w[d]=m1[d]-m0[d];
//     double nw=0.0; for(double v:w) nw+=v*v; nw=std::sqrt(nw);
//     if (nw==0.0){ w.assign(D,0.0); w[0]=1.0; } else
//         for (int64_t d=0; d<D; ++d) w[d]/=nw;

//     // ====== (2) projections & spoke lengths ======  [không tính thời gian]
//     std::vector<double> t(N,0.0), Lspoke(N,0.0);
//     std::vector<double> Xproj(N*D,0.0);
//     for (int64_t i=0;i<N;i++){
//         double ti=0.0;
//         for (int64_t d=0; d<D; ++d) ti += (Xptr[i*D+d]-m0[d])*w[d];
//         t[i]=ti;
//         for (int64_t d=0; d<D; ++d) Xproj[i*D+d]=m0[d]+ti*w[d];
//         double res2=0.0;
//         for (int64_t d=0; d<D; ++d){
//             double r = Xptr[i*D+d]-Xproj[i*D+d];
//             res2 += r*r;
//         }
//         Lspoke[i]=std::sqrt(res2);
//     }

//     // spine order (sort t) — [không tính thời gian]
//     std::vector<int> idx(N); for (int i=0;i<N;i++) idx[i]=i;
//     std::sort(idx.begin(), idx.end(), [&](int a,int b){ return t[a]<t[b]; });

//     // spine cumulative coordinate S — [không tính thời gian]
//     std::vector<double> S(N,0.0);
//     for (int k=1;k<N;k++) S[k]=S[k-1]+std::abs(t[idx[k]]-t[idx[k-1]]);

//     // ====== (3) Build augmented tree (2N nodes) ======  [không tính thời gian]
//     const int TOT = (int)(2*N);
//     std::vector<int> colors(TOT,0);
//     for (int i=0;i<N;i++){ colors[i]=(int)Yptr[i]; colors[N+i]=(int)Yptr[i]; }
//     std::vector<std::tuple<int,int,double>> edges; edges.reserve(3*N);
//     for (int i=0;i<N;i++) edges.emplace_back(i, N+i, Lspoke[i]); // spokes
//     for (int k=0;k<N-1;k++){
//         int a = idx[k], b = idx[k+1];
//         edges.emplace_back(N+a, N+b, std::abs(t[b]-t[a]));        // spine
//     }
//     CSR G = build_csr(TOT, edges);

//     // mapping: spine order -> node id of projection
//     std::vector<int> spine_ids(N);
//     for (int k=0;k<N;k++) spine_ids[k]=N+idx[k];

//     // vị trí projection trên spine: pos[node] in [0..N-1]
//     std::vector<int> pos(TOT, -1);
//     for (int k=0;k<N;k++){
//         int pid = spine_ids[k];
//         pos[pid]=k;
//     }
//     for (int i=0;i<N;i++) pos[i] = pos[N+i];

//     // ====== (4) DP/DFS hai hướng theo từng lớp ======
//     auto compute_side = [&](int root_node, int target_node,
//                             std::vector<double>& out_maxtr, double& tol_root, double& t_side){
//         auto t0 = hrc::now();
//         std::vector<int> parent, depth, order_post; std::vector<double> parent_w;
//         root_tree_iter(G, root_node, parent, parent_w, depth, order_post);
//         std::vector<double> d0,d1,maxtr; double total_noise_root = 0.0;
//         accumulate_noise_on_branch(G, root_node, target_node, order_post, parent, colors,
//                                    d0,d1,maxtr,total_noise_root);
//         auto t1 = hrc::now();
//         out_maxtr.swap(maxtr);
//         tol_root = total_noise_root;
//         t_side = std::chrono::duration_cast<secd>(t1 - t0).count();
//     };

//     // tách spine theo lớp
//     std::vector<int> X0_sorted; X0_sorted.reserve(N);
//     std::vector<int> X1_sorted; X1_sorted.reserve(N);
//     for (int k=0;k<N;k++){
//         int pid = spine_ids[k];
//         if (colors[pid]==0) X0_sorted.push_back(pid); else X1_sorted.push_back(pid);
//     }

//     std::vector<double> max0L(TOT,0.0), max0R(TOT,0.0), max1L(TOT,0.0), max1R(TOT,0.0);
//     double dumpTol=0.0; double t_dpdfs = 0.0;
//     if (!X0_sorted.empty()){
//         int L = X0_sorted.front(), R = X0_sorted.back();
//         double t1=0.0,t2=0.0;
//         compute_side(L, R, max0L, dumpTol, t1); // đi về phải
//         compute_side(R, L, max0R, dumpTol, t2); // đi về trái
//         t_dpdfs += t1 + t2;
//     }
//     if (!X1_sorted.empty()){
//         int L = X1_sorted.front(), R = X1_sorted.back();
//         double t1=0.0,t2=0.0;
//         compute_side(L, R, max1L, dumpTol, t1);
//         compute_side(R, L, max1R, dumpTol, t2);
//         t_dpdfs += t1 + t2;
//     }

//     auto noise_proj = [&](int cls, bool dir_right, int u_proj)->double{
//         if (cls==0) return dir_right ? max0L[u_proj] : max0R[u_proj];
//         else        return dir_right ? max1L[u_proj] : max1R[u_proj];
//     };

//     // f_side(u,v): nhiễu phía cùng nhãn với u khi root=u và đi theo nhánh tới v
//     auto f_side = [&](int u, int v)->double{
//         int cu = colors[u];
//         int iu = pos[u], iv = pos[v];
//         bool dir_right = (iv >= iu);
//         int u_proj = (u < (int)N) ? (N+u) : u;
//         return noise_proj(cu, dir_right, u_proj);
//     };

//     // khoảng cách: spine(u', v') + spoke(u) + spoke(v)
//     auto dist_uv = [&](int u, int v)->double{
//         int iu = pos[u], iv = pos[v];
//         double spine = std::abs(S[iv] - S[iu]);
//         double su = (u < (int)N) ? Lspoke[u] : 0.0;
//         double sv = (v < (int)N) ? Lspoke[v] : 0.0;
//         return spine + su + sv;
//     };

//     // ====== (5) Quét O(n^2) cặp khác nhãn (được đo riêng) ======
//     auto t_pairs_t0 = hrc::now();
//     double best = std::numeric_limits<double>::infinity();
//     int best_u=-1, best_v=-1; double best_d=0.0;

//     for (int u=0; u<(int)TOT; ++u){
//         int cu = colors[u]; if (cu!=0 && cu!=1) continue;
//         for (int v=0; v<(int)TOT; ++v){
//             if (u==v) continue;
//             int cv = colors[v]; if (cv!=0 && cv!=1) continue;
//             if (cu == cv) continue;

//             double f = f_side(u,v) + f_side(v,u);
//             double d = dist_uv(u,v);
//             double L = f - lambda_ * d;
//             if (L < best){ best=L; best_u=u; best_v=v; best_d=d; }
//         }
//     }
//     auto t_pairs_t1 = hrc::now();
//     double t_pairs = std::chrono::duration_cast<secd>(t_pairs_t1 - t_pairs_t0).count();

//     auto to_orig = [&](int node)->int{ return (node < (int)N)? node : (node - (int)N); };
//     bool s_is_proj = (best_u >= (int)N);
//     bool p_is_proj = (best_v >= (int)N);

//     py::dict out;
//     out["s_node"]    = best_u;
//     out["p_node"]    = best_v;
//     out["s_is_proj"] = s_is_proj;
//     out["p_is_proj"] = p_is_proj;
//     out["s_orig"]    = to_orig(best_u);
//     out["p_orig"]    = to_orig(best_v);
//     out["min_val"]   = best;
//     out["dist"]      = best_d;
//     out["lambda"]    = lambda_;
//     // timings
//     out["t_dpdfs"]   = t_dpdfs;           // chỉ DFS + DP (2 hướng x 2 lớp)
//     out["t_pairs"]   = t_pairs;           // O(n^2) scan
//     out["t_fit_core"]= t_dpdfs + t_pairs; // tổng “phần core” yêu cầu
//     return out;
// }

// PYBIND11_MODULE(svm_on_tree_cpp, m){
//     m.doc() = "C++ core for SVM-on-Tree (arbitrary lambda) — DP/DFS timed separately and O(n^2) pair scan timed";
//     m.def("fit_core_lambda_any", &fit_core_lambda_any,
//           py::arg("X"), py::arg("y"), py::arg("lamda")=1.0,
// R"pbdoc(
// Compute optimal (s,p) for arbitrary lambda.
// Timings returned:
//   t_dpdfs   : only DFS + DP passes (both classes, both directions)
//   t_pairs   : O(n^2) scan over opposite-label pairs
//   t_fit_core: t_dpdfs + t_pairs
// )pbdoc");
// }


// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>

// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <chrono>
// #include <cstring>

// namespace py = pybind11;

// using Clock = std::chrono::high_resolution_clock;
// static inline double secs_since(const Clock::time_point& a, const Clock::time_point& b) {
//     return std::chrono::duration<double>(b - a).count();
// }

// // ---------- CSR ----------
// struct CSR {
//     int n;
//     std::vector<long long> indptr;
//     std::vector<int> indices;
//     std::vector<double> weights;
// };

// static CSR build_csr(int n, const std::vector<std::tuple<int,int,double>>& edges) {
//     std::vector<int> deg(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;
//         deg[u]++; deg[v]++;
//     }

//     CSR csr;
//     csr.n = n;
//     csr.indptr.assign(n + 1, 0);
//     for (int i=0;i<n;i++) csr.indptr[i+1] = csr.indptr[i] + deg[i];

//     csr.indices.assign((size_t)csr.indptr.back(), 0);
//     csr.weights.assign((size_t)csr.indptr.back(), 0.0);

//     std::vector<int> fill(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;

//         long long pu = csr.indptr[u] + fill[u]++;
//         csr.indices[(size_t)pu] = v;
//         csr.weights[(size_t)pu] = w;

//         long long pv = csr.indptr[v] + fill[v]++;
//         csr.indices[(size_t)pv] = u;
//         csr.weights[(size_t)pv] = w;
//     }
//     return csr;
// }

// // ---------- Rooting / true postorder (no sort) ----------
// static void root_tree_iter(
//     const CSR& G, int root,
//     std::vector<int>& parent, std::vector<double>& parent_w,
//     std::vector<int>& depth, std::vector<int>& order_post
// ){
//     const int n = G.n;
//     parent.assign(n, -1);
//     parent_w.assign(n, 0.0);
//     depth.assign(n, 0);

//     order_post.clear();
//     order_post.reserve(n);

//     // per-node adjacency iterator
//     std::vector<int> it(n, 0);
//     std::vector<int> st;
//     st.reserve(n);

//     parent[root] = -1;
//     depth[root]  = 0;
//     st.push_back(root);

//     while (!st.empty()) {
//         int u = st.back();
//         int L = (int)G.indptr[u];
//         int R = (int)G.indptr[u+1];

//         if (it[u] < (R - L)) {
//             int k = L + it[u];
//             ++it[u];
//             int v = G.indices[(size_t)k];

//             if (v == parent[u]) continue;     // back edge to parent
//             if (parent[v] != -1) continue;    // already visited

//             parent[v] = u;
//             parent_w[v] = G.weights[(size_t)k];
//             depth[v] = depth[u] + 1;
//             st.push_back(v);
//         } else {
//             order_post.push_back(u);
//             st.pop_back();
//         }
//     }
// }

// // ---------- Helpers ----------
// static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
//     std::vector<uint8_t> contains(parent.size(), 0);
//     int u = target;
//     while (u != -1) {
//         contains[(size_t)u] = 1;
//         u = parent[(size_t)u];
//     }
//     return contains;
// }

// // ---------- sz / dis accumulations ----------
// static void compute_sz(
//     const std::vector<int>& order_post,
//     const std::vector<int>& colors,
//     std::vector<long long>& sz1,
//     std::vector<long long>& sz2
// ){
//     const int n = (int)colors.size();
//     sz1.assign(n, 0);
//     sz2.assign(n, 0);
//     for (int u : order_post) {
//         if (colors[(size_t)u] == 0) sz1[(size_t)u] += 1;
//         else                        sz2[(size_t)u] += 1;
//     }
// }

// static void accumulate_to_parent(
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     std::vector<long long>& sz1,
//     std::vector<long long>& sz2
// ){
//     for (int u : order_post) {
//         int p = parent[(size_t)u];
//         if (p != -1) {
//             sz1[(size_t)p] += sz1[(size_t)u];
//             sz2[(size_t)p] += sz2[(size_t)u];
//         }
//     }
// }

// static void accumulate_noise_on_branch(
//     const CSR& G,
//     int root, int target,
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     const std::vector<int>& colors,
//     const std::vector<long long>& sz1,
//     const std::vector<long long>& sz2,
//     std::vector<double>& dis_s,
//     std::vector<double>& dis_p,
//     std::vector<double>& maxtr,
//     double& total_noise_root
// ){
//     const int n = G.n;
//     std::vector<uint8_t> contains = mark_path_to_target(parent, target);

//     dis_s.assign(n, 0.0);
//     dis_p.assign(n, 0.0);
//     maxtr.assign(n, 0.0);

//     for (int u : order_post) {
//         for (long long k = G.indptr[(size_t)u]; k < G.indptr[(size_t)u + 1]; ++k) {
//             int v = G.indices[(size_t)k];
//             if (parent[(size_t)v] != u) continue;
//             if (!contains[(size_t)v]) continue;

//             double w = G.weights[(size_t)k];
//             dis_s[(size_t)u] += dis_s[(size_t)v] + (double)sz1[(size_t)v] * w;
//             dis_p[(size_t)u] += dis_p[(size_t)v] + (double)sz2[(size_t)v] * w;

//             if (colors[(size_t)u] == colors[(size_t)root]) {
//                 if (colors[(size_t)root] == 0) maxtr[(size_t)u] = dis_s[(size_t)v] + (double)sz1[(size_t)v] * w;
//                 else                           maxtr[(size_t)u] = dis_p[(size_t)v] + (double)sz2[(size_t)v] * w;
//             }
//         }
//     }
//     total_noise_root = (colors[(size_t)root] == 0) ? dis_s[(size_t)root] : dis_p[(size_t)root];
// }

// // ---------- Main core ----------
// py::dict fit_core(py::array_t<double, py::array::c_style | py::array::forcecast> X,
//                   py::array_t<long long, py::array::c_style | py::array::forcecast> y,
//                   double lambda_)
// {
//     auto t_all_start = Clock::now();

//     py::buffer_info bx = X.request();
//     py::buffer_info by = y.request();
//     if (bx.ndim != 2) throw std::runtime_error("X must be 2D");
//     if (by.ndim != 1) throw std::runtime_error("y must be 1D");

//     const int64_t N = bx.shape[0];
//     const int64_t D = bx.shape[1];
//     if (by.shape[0] != N) throw std::runtime_error("X,y length mismatch");
//     if (N < 2) throw std::runtime_error("Need at least 2 samples");

//     const double* Xptr = (const double*)bx.ptr;
//     const long long* Yptr = (const long long*)by.ptr;

//     // ===== Stage A: means/projections =====
//     auto tA0 = Clock::now();

//     std::vector<double> m0((size_t)D, 0.0), m1((size_t)D, 0.0);
//     long long c0=0, c1=0;
//     for (int64_t i=0;i<N;i++){
//         if (Yptr[i]==0){
//             c0++;
//             for (int64_t d=0; d<D; ++d) m0[(size_t)d] += Xptr[(size_t)i*(size_t)D + (size_t)d];
//         } else {
//             c1++;
//             for (int64_t d=0; d<D; ++d) m1[(size_t)d] += Xptr[(size_t)i*(size_t)D + (size_t)d];
//         }
//     }
//     if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 are required.");
//     for (int64_t d=0; d<D; ++d){
//         m0[(size_t)d] /= (double)c0;
//         m1[(size_t)d] /= (double)c1;
//     }

//     std::vector<double> w((size_t)D, 0.0);
//     for (int64_t d=0; d<D; ++d) w[(size_t)d] = m1[(size_t)d] - m0[(size_t)d];
//     double nw=0.0;
//     for (double v: w) nw += v*v;
//     nw = std::sqrt(nw);
//     if (nw==0.0){
//         w[0] = 1.0;
//         for (int64_t d=1; d<D; ++d) w[(size_t)d] = 0.0;
//     } else {
//         for (int64_t d=0; d<D; ++d) w[(size_t)d] /= nw;
//     }

//     std::vector<double> t((size_t)N, 0.0), Lleaf((size_t)N, 0.0);
//     for (int64_t i=0;i<N;i++){
//         double ti = 0.0;
//         for (int64_t d=0; d<D; ++d)
//             ti += (Xptr[(size_t)i*(size_t)D + (size_t)d] - m0[(size_t)d]) * w[(size_t)d];
//         t[(size_t)i] = ti;

//         // residual norm to projection on spine
//         double res2 = 0.0;
//         for (int64_t d=0; d<D; ++d){
//             double proj = m0[(size_t)d] + ti * w[(size_t)d];
//             double r = Xptr[(size_t)i*(size_t)D + (size_t)d] - proj;
//             res2 += r*r;
//         }
//         Lleaf[(size_t)i] = std::sqrt(res2);
//     }

//     auto tA1 = Clock::now();
//     double time_means_proj = secs_since(tA0, tA1);

//     // ===== Stage B: sort along spine =====
//     auto tB0 = Clock::now();

//     std::vector<int> idx((size_t)N);
//     for (int i=0;i<(int)N;i++) idx[(size_t)i] = i;
//     std::sort(idx.begin(), idx.end(), [&](int a, int b){ return t[(size_t)a] < t[(size_t)b]; });

//     std::vector<double> Dsp((size_t)N, 0.0);
//     for (int64_t k=1;k<N;k++){
//         Dsp[(size_t)k] = Dsp[(size_t)(k-1)] + std::abs(t[(size_t)idx[(size_t)k]] - t[(size_t)idx[(size_t)(k-1)]]);
//     }

//     auto tB1 = Clock::now();
//     double time_sort_spine = secs_since(tB0, tB1);

//     // ===== Stage C: build graph =====
//     auto tC0 = Clock::now();

//     const int TOT = (int)(2*N);
//     std::vector<int> colors((size_t)TOT, 0);
//     for (int i=0;i<(int)N;i++){
//         colors[(size_t)i] = (int)Yptr[i];
//         colors[(size_t)(N + i)] = (int)Yptr[i];
//     }

//     std::vector<std::tuple<int,int,double>> edges;
//     edges.reserve((size_t)(3*N));
//     for (int i=0;i<(int)N;i++){
//         edges.emplace_back(i, (int)N + i, Lleaf[(size_t)i]); // leaf -> projection
//     }
//     for (int k=0;k<(int)N-1;k++){
//         int i = idx[(size_t)k];
//         int j = idx[(size_t)(k+1)];
//         double wsp = std::abs(t[(size_t)j] - t[(size_t)i]); // spine edge
//         edges.emplace_back((int)N + i, (int)N + j, wsp);
//     }

//     CSR G = build_csr(TOT, edges);

//     auto tC1 = Clock::now();
//     double time_build_graph = secs_since(tC0, tC1);

//     // spine_ids = projection nodes in sorted order
//     std::vector<int> spine_ids((size_t)N);
//     for (int k=0;k<(int)N;k++) spine_ids[(size_t)k] = (int)N + idx[(size_t)k];

//     // ===== Stage D: DP passes (4 roots) =====
//     auto tD0 = Clock::now();

//     auto compute_side = [&](int root_node, int target_node, std::vector<double>& out_maxtr){
//         std::vector<int> parent, depth, order_post;
//         std::vector<double> parent_w;
//         root_tree_iter(G, root_node, parent, parent_w, depth, order_post);

//         std::vector<long long> sz1, sz2;
//         compute_sz(order_post, colors, sz1, sz2);
//         accumulate_to_parent(order_post, parent, sz1, sz2);

//         std::vector<double> dis_s, dis_p, maxtr;
//         double total_noise_root = 0.0;
//         accumulate_noise_on_branch(G, root_node, target_node, order_post, parent,
//                                    colors, sz1, sz2, dis_s, dis_p, maxtr, total_noise_root);

//         out_maxtr.swap(maxtr);              // avoid copy
//         out_maxtr[(size_t)root_node] = total_noise_root;
//     };

//     std::vector<int> X0_sorted; X0_sorted.reserve((size_t)N);
//     std::vector<int> X1_sorted; X1_sorted.reserve((size_t)N);
//     for (int k=0;k<(int)N;k++){
//         int pid = spine_ids[(size_t)k];
//         if (colors[(size_t)pid] == 0) X0_sorted.push_back(pid);
//         else                          X1_sorted.push_back(pid);
//     }

//     std::vector<double> maxtr_X0_left((size_t)TOT, 0.0), maxtr_X0_right((size_t)TOT, 0.0),
//                         maxtr_X1_left((size_t)TOT, 0.0), maxtr_X1_right((size_t)TOT, 0.0);

//     if (X0_sorted.size() >= 2) {
//         compute_side(X0_sorted.front(), X0_sorted.back(),  maxtr_X0_left);
//         compute_side(X0_sorted.back(),  X0_sorted.front(), maxtr_X0_right);
//     } else if (X0_sorted.size() == 1) {
//         compute_side(X0_sorted.front(), X0_sorted.front(), maxtr_X0_left);
//         maxtr_X0_right = maxtr_X0_left;
//     }

//     if (X1_sorted.size() >= 2) {
//         compute_side(X1_sorted.front(), X1_sorted.back(),  maxtr_X1_left);
//         compute_side(X1_sorted.back(),  X1_sorted.front(), maxtr_X1_right);
//     } else if (X1_sorted.size() == 1) {
//         compute_side(X1_sorted.front(), X1_sorted.front(), maxtr_X1_left);
//         maxtr_X1_right = maxtr_X1_left;
//     }

//     auto tD1 = Clock::now();
//     double time_dp = secs_since(tD0, tD1);

//     // ===== Stage E: scan adjacent boundary pairs =====
//     auto tE0 = Clock::now();

//     double best = std::numeric_limits<double>::infinity();
//     int best_s = -1, best_p = -1;

//     for (int k=0;k<(int)N-1;k++){
//         int s = spine_ids[(size_t)k];
//         int p = spine_ids[(size_t)(k+1)];
//         if (colors[(size_t)s] == colors[(size_t)p]) continue;

//         double dist_sp = Dsp[(size_t)(k+1)] - Dsp[(size_t)k];
//         double ans = 0.0;

//         if (colors[(size_t)s] == 0) {
//             ans = maxtr_X0_left[(size_t)s] + maxtr_X1_right[(size_t)p] - lambda_ * dist_sp;
//         } else {
//             ans = maxtr_X1_left[(size_t)s] + maxtr_X0_right[(size_t)p] - lambda_ * dist_sp;
//         }

//         if (ans < best) {
//             best = ans;
//             best_s = s;
//             best_p = p;
//         }
//     }

//     auto tE1 = Clock::now();
//     double time_scan_pairs = secs_since(tE0, tE1);

//     // map projection node -> original sample index
//     int support_s_idx = (best_s >= (int)N) ? (best_s - (int)N) : best_s;
//     int support_p_idx = (best_p >= (int)N) ? (best_p - (int)N) : best_p;

//     // export prediction params (for fair/fast predict)
//     double ts = t[(size_t)support_s_idx];
//     double tp = t[(size_t)support_p_idx];
//     double thr = 0.5 * (ts + tp);

//     long long ys = Yptr[support_s_idx];
//     long long yp = Yptr[support_p_idx];
//     long long y_left, y_right;
//     if (ts <= tp) { y_left = ys; y_right = yp; }
//     else          { y_left = yp; y_right = ys; }

//     auto t_all_end = Clock::now();
//     double time_total = secs_since(t_all_start, t_all_end);
//     double time_no_spine_sort = time_total - time_sort_spine;

//     // numpy arrays for m0 and w_unit
//     py::array_t<double> m0_arr((py::ssize_t)D);
//     py::array_t<double> w_arr((py::ssize_t)D);
//     std::memcpy((double*)m0_arr.mutable_data(), m0.data(), (size_t)D * sizeof(double));
//     std::memcpy((double*)w_arr.mutable_data(),  w.data(),  (size_t)D * sizeof(double));

//     py::dict out;
//     out["support_s"] = support_s_idx;
//     out["support_p"] = support_p_idx;
//     out["min_val"]   = best;

//     out["m0"]        = m0_arr;
//     out["w_unit"]    = w_arr;
//     out["thr"]       = thr;
//     out["y_left"]    = (long long)y_left;
//     out["y_right"]   = (long long)y_right;

//     out["time_total"]         = time_total;
//     out["time_means_proj"]    = time_means_proj;
//     out["time_sort_spine"]    = time_sort_spine;
//     out["time_build_graph"]   = time_build_graph;
//     out["time_dp"]            = time_dp;
//     out["time_scan_pairs"]    = time_scan_pairs;
//     out["time_no_spine_sort"] = time_no_spine_sort;

//     return out;
// }

// // PYBIND11_MODULE(svm_on_tree_cpp, m) {
// //     m.doc() = "C++ core for SVM On Tree (pybind11, timed + exports prediction params)";
// //     m.def("fit_core", &fit_core, py::arg("X"), py::arg("y"), py::arg("lambda_")=1.0,
// //           R"pbdoc(
// //             Core routine for SVM On Tree.
// //             Returns:
// //               support_s, support_p, min_val,
// //               m0, w_unit, thr, y_left, y_right,
// //               time_total, time_means_proj, time_sort_spine, time_build_graph,
// //               time_dp, time_scan_pairs, time_no_spine_sort
// //           )pbdoc");
// //     m.attr("BUILD_TAG") = "root_postorder_no_sort_v2_export_pred";
// // }

// PYBIND11_MODULE(svm_on_tree_cpp, m) {
//     m.def("fit_core", &fit_core, py::arg("X"), py::arg("y"), py::arg("lambda_")=1.0);
//     m.attr("BUILD_TAG") = "root_postorder_no_sort_v2_export_pred";
// }


// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <chrono>

// namespace py = pybind11;
// using Clock = std::chrono::high_resolution_clock;

// static inline double secs_since(const Clock::time_point& a, const Clock::time_point& b) {
//     return std::chrono::duration<double>(b - a).count();
// }

// // ---------- CSR builder ----------
// struct CSR {
//     int n;
//     std::vector<long long> indptr;
//     std::vector<int> indices;
//     std::vector<double> weights;
// };

// static CSR build_csr(int n, const std::vector<std::tuple<int,int,double>>& edges) {
//     std::vector<int> deg(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;
//         deg[u]++; deg[v]++;
//     }
//     CSR csr;
//     csr.n = n;
//     csr.indptr.assign(n + 1, 0);
//     for (int i=0;i<n;i++) csr.indptr[i+1] = csr.indptr[i] + deg[i];
//     csr.indices.assign(csr.indptr.back(), 0);
//     csr.weights.assign(csr.indptr.back(), 0.0);

//     std::vector<int> fill(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;

//         long long pu = csr.indptr[u] + fill[u];
//         csr.indices[pu] = v;
//         csr.weights[pu] = w;
//         fill[u]++;

//         long long pv = csr.indptr[v] + fill[v];
//         csr.indices[pv] = u;
//         csr.weights[pv] = w;
//         fill[v]++;
//     }
//     return csr;
// }

// // ---------- Rooting / orders (true postorder, no sort, no revisits) ----------
// static void root_tree_iter(
//     const CSR& G, int root,
//     std::vector<int>& parent, std::vector<double>& parent_w,
//     std::vector<int>& depth, std::vector<int>& order_post
// ){
//     const int n = G.n;
//     parent.assign(n, -1);
//     parent_w.assign(n, 0.0);
//     depth.assign(n, 0);
//     order_post.clear();
//     order_post.reserve(n);

//     std::vector<int> it(n, 0);
//     std::vector<int> stack;
//     stack.reserve(n);

//     parent[root] = -1;
//     depth[root] = 0;
//     stack.push_back(root);

//     while (!stack.empty()){
//         int u = stack.back();
//         int L = static_cast<int>(G.indptr[u]);
//         int R = static_cast<int>(G.indptr[u+1]);

//         if (it[u] < (R - L)) {
//             int k = L + it[u];
//             ++it[u];
//             int v = G.indices[k];

//             if (v == parent[u]) continue;      // don't go back to parent
//             if (parent[v] != -1) continue;     // already visited -> skip

//             parent[v] = u;
//             parent_w[v] = G.weights[k];
//             depth[v] = depth[u] + 1;
//             stack.push_back(v);
//         } else {
//             order_post.push_back(u);
//             stack.pop_back();
//         }
//     }
// }

// static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
//     std::vector<uint8_t> contains(parent.size(), 0);
//     int u = target;
//     while (u != -1) {
//         contains[u] = 1;
//         u = parent[u];
//     }
//     return contains;
// }

// static void compute_sz(
//     const std::vector<int>& order_post,
//     const std::vector<int>& colors,
//     std::vector<long long>& sz1,
//     std::vector<long long>& sz2
// ){
//     const int n = static_cast<int>(colors.size());
//     sz1.assign(n, 0);
//     sz2.assign(n, 0);
//     for (int u : order_post) {
//         if (colors[u] == 0) sz1[u] += 1;
//         else                sz2[u] += 1;
//     }
// }

// static void accumulate_to_parent(
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     std::vector<long long>& sz1,
//     std::vector<long long>& sz2
// ){
//     for (int u : order_post) {
//         int p = parent[u];
//         if (p != -1){
//             sz1[p] += sz1[u];
//             sz2[p] += sz2[u];
//         }
//     }
// }

// static void accumulate_noise_on_branch(
//     const CSR& G,
//     int root, int target,
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     const std::vector<int>& colors,
//     const std::vector<long long>& sz1,
//     const std::vector<long long>& sz2,
//     std::vector<double>& dis_s,
//     std::vector<double>& dis_p,
//     std::vector<double>& maxtr,
//     double& total_noise_root
// ){
//     const int n = G.n;
//     std::vector<uint8_t> contains = mark_path_to_target(parent, target);
//     dis_s.assign(n, 0.0);
//     dis_p.assign(n, 0.0);
//     maxtr.assign(n, 0.0);

//     for (int u : order_post) {
//         for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
//             int v = G.indices[k];
//             if (parent[v] != u) continue;
//             if (!contains[v]) continue;

//             double w = G.weights[k];
//             dis_s[u] += dis_s[v] + (double)sz1[v] * w;
//             dis_p[u] += dis_p[v] + (double)sz2[v] * w;

//             if (colors[u] == colors[root]){
//                 if (colors[root] == 0) maxtr[u] = dis_s[v] + (double)sz1[v] * w;
//                 else                   maxtr[u] = dis_p[v] + (double)sz2[v] * w;
//             }
//         }
//     }
//     total_noise_root = (colors[root] == 0) ? dis_s[root] : dis_p[root];
// }

// py::dict fit_core(py::array_t<double, py::array::c_style | py::array::forcecast> X,
//                   py::array_t<long long, py::array::c_style | py::array::forcecast> y,
//                   double lambda_)
// {
//     auto t_all_start = Clock::now();

//     py::buffer_info bx = X.request();
//     py::buffer_info by = y.request();
//     if (bx.ndim != 2) throw std::runtime_error("X must be 2D");
//     if (by.ndim != 1) throw std::runtime_error("y must be 1D");
//     const int64_t N = bx.shape[0];
//     const int64_t D = bx.shape[1];
//     if (by.shape[0] != N) throw std::runtime_error("X,y length mismatch");
//     if (N < 2) throw std::runtime_error("Need at least 2 samples");

//     const double* Xptr = static_cast<double*>(bx.ptr);
//     const long long* Yptr = static_cast<long long*>(by.ptr);

//     // ===== Stage A: means/projections =====
//     auto tA0 = Clock::now();
//     std::vector<double> m0(D, 0.0), m1(D, 0.0);
//     long long c0 = 0, c1 = 0;
//     for (int64_t i=0;i<N;i++){
//         if (Yptr[i]==0){ c0++; for (int64_t d=0; d<D; ++d) m0[d]+=Xptr[i*D+d]; }
//         else            { c1++; for (int64_t d=0; d<D; ++d) m1[d]+=Xptr[i*D+d]; }
//     }
//     if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 are required.");
//     for (int64_t d=0; d<D; ++d){ m0[d]/= (double)c0; m1[d]/= (double)c1; }

//     std::vector<double> w(D, 0.0);
//     for (int64_t d=0; d<D; ++d) w[d] = m1[d] - m0[d];
//     double nw=0.0; for (double v:w) nw += v*v; nw = std::sqrt(nw);
//     if (nw==0.0){ w[0]=1.0; for (int64_t d=1; d<D; ++d) w[d]=0.0; }
//     else        { for (int64_t d=0; d<D; ++d) w[d]/=nw; }

//     std::vector<double> t(N, 0.0), Lleaf(N, 0.0);
//     for (int64_t i=0;i<N;i++){
//         double ti = 0.0;
//         for (int64_t d=0; d<D; ++d) ti += (Xptr[i*D+d] - m0[d])*w[d];
//         t[i] = ti;

//         double res2 = 0.0;
//         for (int64_t d=0; d<D; ++d){
//             double proj = m0[d] + ti*w[d];
//             double r = Xptr[i*D+d] - proj;
//             res2 += r*r;
//         }
//         Lleaf[i] = std::sqrt(res2);
//     }
//     auto tA1 = Clock::now();
//     double time_means_proj = secs_since(tA0, tA1);

//     // ===== Stage B: sort along spine =====
//     auto tB0 = Clock::now();
//     std::vector<int> idx(N); for (int i=0;i<N;i++) idx[i]=i;
//     std::sort(idx.begin(), idx.end(), [&](int a, int b){ return t[a] < t[b]; });

//     std::vector<double> Dsp(N, 0.0);
//     for (int k=1;k<N;k++){
//         Dsp[k] = Dsp[k-1] + std::abs(t[idx[k]] - t[idx[k-1]]);
//     }
//     auto tB1 = Clock::now();
//     double time_sort_spine = secs_since(tB0, tB1);

//     // ===== Stage C: build graph (CSR) =====
//     auto tC0 = Clock::now();
//     const int TOT = (int)(2*N);
//     std::vector<int> colors(TOT, 0);
//     for (int i=0;i<N;i++){
//         colors[i] = (int)Yptr[i];
//         colors[N+i] = (int)Yptr[i];
//     }

//     std::vector<std::tuple<int,int,double>> edges;
//     edges.reserve(3*N);
//     for (int i=0;i<N;i++){
//         edges.emplace_back(i, N+i, Lleaf[i]);
//     }
//     for (int k=0;k<N-1;k++){
//         int i = idx[k], j = idx[k+1];
//         double wsp = std::abs(t[j]-t[i]);
//         edges.emplace_back(N+i, N+j, wsp);
//     }
//     CSR G = build_csr(TOT, edges);
//     auto tC1 = Clock::now();
//     double time_build_graph = secs_since(tC0, tC1);

//     std::vector<int> spine_ids(N);
//     for (int k=0;k<N;k++) spine_ids[k] = N + idx[k];

//     // ===== Stage D: DP/DFS passes =====
//     auto tD0 = Clock::now();

//     auto compute_side = [&](int root_node, int target_node, std::vector<double>& out_maxtr){
//         std::vector<int> parent; std::vector<double> parent_w; std::vector<int> depth; std::vector<int> order_post;
//         root_tree_iter(G, root_node, parent, parent_w, depth, order_post);

//         std::vector<long long> sz1, sz2;
//         compute_sz(order_post, colors, sz1, sz2);
//         accumulate_to_parent(order_post, parent, sz1, sz2);

//         std::vector<double> dis_s, dis_p, maxtr;
//         double total_noise_root = 0.0;
//         accumulate_noise_on_branch(G, root_node, target_node, order_post, parent, colors, sz1, sz2,
//                                    dis_s, dis_p, maxtr, total_noise_root);
//         out_maxtr = maxtr;
//         out_maxtr[root_node] = total_noise_root;
//     };

//     std::vector<int> X0_sorted; X0_sorted.reserve(N);
//     std::vector<int> X1_sorted; X1_sorted.reserve(N);
//     for (int k=0;k<N;k++){
//         int pid = spine_ids[k];
//         if (colors[pid]==0) X0_sorted.push_back(pid);
//         else                X1_sorted.push_back(pid);
//     }

//     std::vector<double> maxtr_X0_left(TOT, 0.0), maxtr_X0_right(TOT, 0.0),
//                         maxtr_X1_left(TOT, 0.0), maxtr_X1_right(TOT, 0.0);

//     if (X0_sorted.size()>=2){
//         compute_side(X0_sorted.front(), X0_sorted.back(),  maxtr_X0_left);
//         compute_side(X0_sorted.back(),  X0_sorted.front(), maxtr_X0_right);
//     } else if (X0_sorted.size()==1){
//         compute_side(X0_sorted.front(), X0_sorted.front(), maxtr_X0_left);
//         maxtr_X0_right = maxtr_X0_left;
//     }

//     if (X1_sorted.size()>=2){
//         compute_side(X1_sorted.front(), X1_sorted.back(),  maxtr_X1_left);
//         compute_side(X1_sorted.back(),  X1_sorted.front(), maxtr_X1_right);
//     } else if (X1_sorted.size()==1){
//         compute_side(X1_sorted.front(), X1_sorted.front(), maxtr_X1_left);
//         maxtr_X1_right = maxtr_X1_left;
//     }

//     auto tD1 = Clock::now();
//     double time_dp = secs_since(tD0, tD1);

//     // ===== Stage E: scan adjacent pairs =====
//     auto tE0 = Clock::now();
//     double best = std::numeric_limits<double>::infinity();
//     int best_s=-1, best_p=-1;
//     for (int k=0;k<N-1;k++){
//         int s = spine_ids[k];
//         int p = spine_ids[k+1];
//         if (colors[s]==colors[p]) continue;

//         double dist_sp = Dsp[k+1]-Dsp[k];
//         double ans = 0.0;
//         if (colors[s]==0){
//             ans = maxtr_X0_left[s] + maxtr_X1_right[p] - lambda_ * dist_sp;
//         } else {
//             ans = maxtr_X1_left[s] + maxtr_X0_right[p] - lambda_ * dist_sp;
//         }
//         if (ans < best){
//             best = ans; best_s = s; best_p = p;
//         }
//     }
//     auto tE1 = Clock::now();
//     double time_scan_pairs = secs_since(tE0, tE1);

//     int support_s_idx = (best_s>= (int)N) ? (best_s - (int)N) : best_s;
//     int support_p_idx = (best_p>= (int)N) ? (best_p - (int)N) : best_p;

//     auto t_all_end = Clock::now();
//     double time_total = secs_since(t_all_start, t_all_end);
//     double time_no_spine_sort = time_total - time_sort_spine;

//     py::dict out;
//     out["support_s"] = support_s_idx;
//     out["support_p"] = support_p_idx;
//     out["min_val"]   = best;

//     out["time_total"]         = time_total;
//     out["time_means_proj"]    = time_means_proj;
//     out["time_sort_spine"]    = time_sort_spine;
//     out["time_build_graph"]   = time_build_graph;
//     out["time_dp"]            = time_dp;
//     out["time_scan_pairs"]    = time_scan_pairs;
//     out["time_no_spine_sort"] = time_no_spine_sort;

//     return out;
// }

// PYBIND11_MODULE(svm_on_tree_cpp, m) {
//     m.doc() = "SVM On Tree C++ core (pybind11, timed stages)";
//     m.def("fit_core", &fit_core, py::arg("X"), py::arg("y"), py::arg("lambda_")=1.0);
//     m.attr("BUILD_TAG") = "any_lambda_export_fit_core_v1";
// }

//! FOURTH

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <chrono>

// namespace py = pybind11;
// using Clock = std::chrono::high_resolution_clock;

// static inline double secs_since(const Clock::time_point& a, const Clock::time_point& b) {
//     return std::chrono::duration<double>(b - a).count();
// }

// // ---------- CSR builder ----------
// struct CSR {
//     int n;
//     std::vector<long long> indptr;
//     std::vector<int> indices;
//     std::vector<double> weights;
// };

// static CSR build_csr(int n, const std::vector<std::tuple<int,int,double>>& edges) {
//     std::vector<int> deg(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;
//         deg[u]++; deg[v]++;
//     }
//     CSR csr;
//     csr.n = n;
//     csr.indptr.assign(n + 1, 0);
//     for (int i=0;i<n;i++) csr.indptr[i+1] = csr.indptr[i] + deg[i];
//     csr.indices.assign(csr.indptr.back(), 0);
//     csr.weights.assign(csr.indptr.back(), 0.0);

//     std::vector<int> fill(n, 0);
//     for (auto &e : edges) {
//         int u, v; double w;
//         std::tie(u, v, w) = e;

//         long long pu = csr.indptr[u] + fill[u];
//         csr.indices[pu] = v;
//         csr.weights[pu] = w;
//         fill[u]++;

//         long long pv = csr.indptr[v] + fill[v];
//         csr.indices[pv] = u;
//         csr.weights[pv] = w;
//         fill[v]++;
//     }
//     return csr;
// }

// // ---------- Rooting / orders (true postorder, no sort, no revisits) ----------
// static void root_tree_iter(
//     const CSR& G, int root,
//     std::vector<int>& parent, std::vector<double>& parent_w,
//     std::vector<int>& depth, std::vector<int>& order_post
// ){
//     const int n = G.n;
//     parent.assign(n, -1);
//     parent_w.assign(n, 0.0);
//     depth.assign(n, 0);
//     order_post.clear();
//     order_post.reserve(n);

//     std::vector<int> it(n, 0);
//     std::vector<int> stack;
//     stack.reserve(n);

//     parent[root] = -1;
//     depth[root] = 0;
//     stack.push_back(root);

//     while (!stack.empty()){
//         int u = stack.back();
//         int L = static_cast<int>(G.indptr[u]);
//         int R = static_cast<int>(G.indptr[u+1]);

//         if (it[u] < (R - L)) {
//             int k = L + it[u];
//             ++it[u];
//             int v = G.indices[k];

//             if (v == parent[u]) continue;      // don't go back to parent
//             if (parent[v] != -1) continue;     // already visited -> skip

//             parent[v] = u;
//             parent_w[v] = G.weights[k];
//             depth[v] = depth[u] + 1;
//             stack.push_back(v);
//         } else {
//             order_post.push_back(u);
//             stack.pop_back();
//         }
//     }
// }

// static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
//     std::vector<uint8_t> contains(parent.size(), 0);
//     int u = target;
//     while (u != -1) {
//         contains[u] = 1;
//         u = parent[u];
//     }
//     return contains;
// }

// static void compute_sz(
//     const std::vector<int>& order_post,
//     const std::vector<int>& colors,
//     std::vector<long long>& sz1,
//     std::vector<long long>& sz2
// ){
//     const int n = static_cast<int>(colors.size());
//     sz1.assign(n, 0);
//     sz2.assign(n, 0);
//     for (int u : order_post) {
//         if (colors[u] == 0) sz1[u] += 1;
//         else                sz2[u] += 1;
//     }
// }

// static void accumulate_to_parent(
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     std::vector<long long>& sz1,
//     std::vector<long long>& sz2
// ){
//     for (int u : order_post) {
//         int p = parent[u];
//         if (p != -1){
//             sz1[p] += sz1[u];
//             sz2[p] += sz2[u];
//         }
//     }
// }

// static void accumulate_noise_on_branch(
//     const CSR& G,
//     int root, int target,
//     const std::vector<int>& order_post,
//     const std::vector<int>& parent,
//     const std::vector<int>& colors,
//     const std::vector<long long>& sz1,
//     const std::vector<long long>& sz2,
//     std::vector<double>& dis_s,
//     std::vector<double>& dis_p,
//     std::vector<double>& maxtr,
//     double& total_noise_root
// ){
//     const int n = G.n;
//     std::vector<uint8_t> contains = mark_path_to_target(parent, target);
//     dis_s.assign(n, 0.0);
//     dis_p.assign(n, 0.0);
//     maxtr.assign(n, 0.0);

//     for (int u : order_post) {
//         for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
//             int v = G.indices[k];
//             if (parent[v] != u) continue;
//             if (!contains[v]) continue;

//             double w = G.weights[k];
//             dis_s[u] += dis_s[v] + (double)sz1[v] * w;
//             dis_p[u] += dis_p[v] + (double)sz2[v] * w;

//             if (colors[u] == colors[root]){
//                 if (colors[root] == 0) maxtr[u] = dis_s[v] + (double)sz1[v] * w;
//                 else                   maxtr[u] = dis_p[v] + (double)sz2[v] * w;
//             }
//         }
//     }
//     total_noise_root = (colors[root] == 0) ? dis_s[root] : dis_p[root];
// }

// py::dict fit_core(py::array_t<double, py::array::c_style | py::array::forcecast> X,
//                   py::array_t<long long, py::array::c_style | py::array::forcecast> y,
//                   double lambda_)
// {
//     auto t_all_start = Clock::now();

//     py::buffer_info bx = X.request();
//     py::buffer_info by = y.request();
//     if (bx.ndim != 2) throw std::runtime_error("X must be 2D");
//     if (by.ndim != 1) throw std::runtime_error("y must be 1D");
//     const int64_t N = bx.shape[0];
//     const int64_t D = bx.shape[1];
//     if (by.shape[0] != N) throw std::runtime_error("X,y length mismatch");
//     if (N < 2) throw std::runtime_error("Need at least 2 samples");

//     const double* Xptr = static_cast<double*>(bx.ptr);
//     const long long* Yptr = static_cast<long long*>(by.ptr);

//     // ===== Stage A: means/projections =====
//     auto tA0 = Clock::now();
//     std::vector<double> m0(D, 0.0), m1(D, 0.0);
//     long long c0 = 0, c1 = 0;
//     for (int64_t i=0;i<N;i++){
//         if (Yptr[i]==0){ c0++; for (int64_t d=0; d<D; ++d) m0[d]+=Xptr[i*D+d]; }
//         else            { c1++; for (int64_t d=0; d<D; ++d) m1[d]+=Xptr[i*D+d]; }
//     }
//     if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 are required.");
//     for (int64_t d=0; d<D; ++d){ m0[d]/= (double)c0; m1[d]/= (double)c1; }

//     std::vector<double> w(D, 0.0);
//     for (int64_t d=0; d<D; ++d) w[d] = m1[d] - m0[d];
//     double nw=0.0; for (double v:w) nw += v*v; nw = std::sqrt(nw);
//     if (nw==0.0){ w[0]=1.0; for (int64_t d=1; d<D; ++d) w[d]=0.0; }
//     else        { for (int64_t d=0; d<D; ++d) w[d]/=nw; }

//     std::vector<double> t(N, 0.0), Lleaf(N, 0.0);
//     for (int64_t i=0;i<N;i++){
//         double ti = 0.0;
//         for (int64_t d=0; d<D; ++d) ti += (Xptr[i*D+d] - m0[d])*w[d];
//         t[i] = ti;

//         double res2 = 0.0;
//         for (int64_t d=0; d<D; ++d){
//             double proj = m0[d] + ti*w[d];
//             double r = Xptr[i*D+d] - proj;
//             res2 += r*r;
//         }
//         Lleaf[i] = std::sqrt(res2);
//     }
//     auto tA1 = Clock::now();
//     double time_means_proj = secs_since(tA0, tA1);

//     // ===== Stage B: sort along spine =====
//     auto tB0 = Clock::now();
//     std::vector<int> idx(N); for (int i=0;i<(int)N;i++) idx[i]=i;
//     std::sort(idx.begin(), idx.end(), [&](int a, int b){ return t[a] < t[b]; });

//     std::vector<double> Dsp(N, 0.0);
//     for (int k=1;k<(int)N;k++){
//         Dsp[k] = Dsp[k-1] + std::abs(t[idx[k]] - t[idx[k-1]]);
//     }
//     auto tB1 = Clock::now();
//     double time_sort_spine = secs_since(tB0, tB1);

//     // ===== Stage C: build graph (CSR) =====
//     auto tC0 = Clock::now();
//     const int TOT = (int)(2*N);
//     std::vector<int> colors(TOT, 0);
//     for (int i=0;i<(int)N;i++){
//         colors[i] = (int)Yptr[i];
//         colors[(int)N+i] = (int)Yptr[i];
//     }

//     std::vector<std::tuple<int,int,double>> edges;
//     edges.reserve((size_t)3*N);
//     for (int i=0;i<(int)N;i++){
//         edges.emplace_back(i, (int)N+i, Lleaf[i]);
//     }
//     for (int k=0;k<(int)N-1;k++){
//         int i = idx[k], j = idx[k+1];
//         double wsp = std::abs(t[j]-t[i]);
//         edges.emplace_back((int)N+i, (int)N+j, wsp);
//     }
//     CSR G = build_csr(TOT, edges);
//     auto tC1 = Clock::now();
//     double time_build_graph = secs_since(tC0, tC1);

//     std::vector<int> spine_ids(N);
//     for (int k=0;k<(int)N;k++) spine_ids[k] = (int)N + idx[k];

//     // ===== Stage D: DP/DFS passes =====
//     auto tD0 = Clock::now();

//     auto compute_side = [&](int root_node, int target_node, std::vector<double>& out_maxtr){
//         std::vector<int> parent; std::vector<double> parent_w; std::vector<int> depth; std::vector<int> order_post;
//         root_tree_iter(G, root_node, parent, parent_w, depth, order_post);

//         std::vector<long long> sz1, sz2;
//         compute_sz(order_post, colors, sz1, sz2);
//         accumulate_to_parent(order_post, parent, sz1, sz2);

//         std::vector<double> dis_s, dis_p, maxtr;
//         double total_noise_root = 0.0;
//         accumulate_noise_on_branch(G, root_node, target_node, order_post, parent, colors, sz1, sz2,
//                                    dis_s, dis_p, maxtr, total_noise_root);
//         out_maxtr = maxtr;
//         out_maxtr[root_node] = total_noise_root;
//     };

//     std::vector<int> X0_sorted; X0_sorted.reserve((size_t)N);
//     std::vector<int> X1_sorted; X1_sorted.reserve((size_t)N);
//     for (int k=0;k<(int)N;k++){
//         int pid = spine_ids[k];
//         if (colors[pid]==0) X0_sorted.push_back(pid);
//         else                X1_sorted.push_back(pid);
//     }

//     std::vector<double> maxtr_X0_left(TOT, 0.0), maxtr_X0_right(TOT, 0.0),
//                         maxtr_X1_left(TOT, 0.0), maxtr_X1_right(TOT, 0.0);

//     if (X0_sorted.size()>=2){
//         compute_side(X0_sorted.front(), X0_sorted.back(),  maxtr_X0_left);   // direction to the right
//         compute_side(X0_sorted.back(),  X0_sorted.front(), maxtr_X0_right);  // direction to the left
//     } else if (X0_sorted.size()==1){
//         compute_side(X0_sorted.front(), X0_sorted.front(), maxtr_X0_left);
//         maxtr_X0_right = maxtr_X0_left;
//     }

//     if (X1_sorted.size()>=2){
//         compute_side(X1_sorted.front(), X1_sorted.back(),  maxtr_X1_left);   // direction to the right
//         compute_side(X1_sorted.back(),  X1_sorted.front(), maxtr_X1_right);  // direction to the left
//     } else if (X1_sorted.size()==1){
//         compute_side(X1_sorted.front(), X1_sorted.front(), maxtr_X1_left);
//         maxtr_X1_right = maxtr_X1_left;
//     }

//     auto tD1 = Clock::now();
//     double time_dp = secs_since(tD0, tD1);

//     // ===== Stage E: scan pairs =====
//     // lambda ~= 1  -> O(n) adjacent scan (as before)
//     // lambda != 1  -> O(n^2) scan all opposite-label spine pairs
//     auto tE0 = Clock::now();

//     double best = std::numeric_limits<double>::infinity();
//     int best_left=-1, best_right=-1; // store endpoints in spine order (left pos < right pos)

//     const double eps = 1e-12;
//     const bool lambda_is_one = (std::abs(lambda_ - 1.0) <= eps);

//     if (lambda_is_one) {
//         // O(n) adjacent scan
//         for (int k=0;k<(int)N-1;k++){
//             int a = spine_ids[k];
//             int b = spine_ids[k+1];
//             if (colors[a]==colors[b]) continue;

//             double dist_sp = Dsp[k+1]-Dsp[k];
//             double ans;

//             // If left endpoint is class 0 and right is class 1:
//             //    f = X0_left[a] + X1_right[b]
//             // else (left is class 1, right is class 0):
//             //    ordered (u=class0 at right, v=class1 at left) => f = X0_right[b] + X1_left[a]
//             if (colors[a]==0) {
//                 ans = maxtr_X0_left[a] + maxtr_X1_right[b] - lambda_ * dist_sp;
//             } else {
//                 ans = maxtr_X0_right[b] + maxtr_X1_left[a] - lambda_ * dist_sp;
//             }

//             if (ans < best){
//                 best = ans;
//                 best_left = a;
//                 best_right = b;
//             }
//         }
//     } else {
//         // O(n^2) scan all opposite-label spine pairs
//         for (int i=0;i<(int)N-1;i++){
//             int a = spine_ids[i];
//             int ca = colors[a];

//             for (int j=i+1;j<(int)N;j++){
//                 int b = spine_ids[j];
//                 int cb = colors[b];
//                 if (ca == cb) continue;

//                 double dist_sp = Dsp[j] - Dsp[i];
//                 double ans;

//                 if (ca == 0 && cb == 1) {
//                     // left is class0, right is class1
//                     ans = maxtr_X0_left[a] + maxtr_X1_right[b] - lambda_ * dist_sp;
//                 } else {
//                     // left is class1, right is class0
//                     // ordered (u=class0 at right b, v=class1 at left a):
//                     ans = maxtr_X0_right[b] + maxtr_X1_left[a] - lambda_ * dist_sp;
//                 }

//                 if (ans < best){
//                     best = ans;
//                     best_left = a;
//                     best_right = b;
//                 }
//             }
//         }
//     }

//     auto tE1 = Clock::now();
//     double time_scan_pairs = secs_since(tE0, tE1);

//     // Convert chosen spine node ids back to original indices [0..N-1]
//     int support_s_idx = (best_left  >= (int)N) ? (best_left  - (int)N) : best_left;
//     int support_p_idx = (best_right >= (int)N) ? (best_right - (int)N) : best_right;

//     auto t_all_end = Clock::now();
//     double time_total = secs_since(t_all_start, t_all_end);
//     double time_no_spine_sort = time_total - time_sort_spine;

//     py::dict out;
//     out["support_s"] = support_s_idx;
//     out["support_p"] = support_p_idx;
//     out["min_val"]   = best;

//     out["time_total"]         = time_total;
//     out["time_means_proj"]    = time_means_proj;
//     out["time_sort_spine"]    = time_sort_spine;
//     out["time_build_graph"]   = time_build_graph;
//     out["time_dp"]            = time_dp;
//     out["time_scan_pairs"]    = time_scan_pairs;
//     out["time_no_spine_sort"] = time_no_spine_sort;

//     return out;
// }

// PYBIND11_MODULE(svm_on_tree_cpp, m) {
//     m.doc() = "SVM On Tree C++ core (pybind11, timed stages)";
//     m.def("fit_core", &fit_core, py::arg("X"), py::arg("y"), py::arg("lambda_")=1.0);
//     m.attr("BUILD_TAG") = "any_lambda_export_fit_core_v2_allpairs_if_lambda_ne_1";
// }

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <tuple>
#include <cstdint>

namespace py = pybind11;
using Clock = std::chrono::high_resolution_clock;

static inline double secs_since(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double>(b - a).count();
}

// ---------- CSR builder ----------
struct CSR {
    int n;
    std::vector<long long> indptr;
    std::vector<int> indices;
    std::vector<double> weights;
};

static CSR build_csr(int n, const std::vector<std::tuple<int,int,double>>& edges) {
    std::vector<int> deg(n, 0);
    for (auto &e : edges) {
        int u, v; double w;
        std::tie(u, v, w) = e;
        deg[u]++; deg[v]++;
    }
    CSR csr;
    csr.n = n;
    csr.indptr.assign(n + 1, 0);
    for (int i=0;i<n;i++) csr.indptr[i+1] = csr.indptr[i] + deg[i];
    csr.indices.assign(csr.indptr.back(), 0);
    csr.weights.assign(csr.indptr.back(), 0.0);

    std::vector<int> fill(n, 0);
    for (auto &e : edges) {
        int u, v; double w;
        std::tie(u, v, w) = e;

        long long pu = csr.indptr[u] + fill[u];
        csr.indices[pu] = v;
        csr.weights[pu] = w;
        fill[u]++;

        long long pv = csr.indptr[v] + fill[v];
        csr.indices[pv] = u;
        csr.weights[pv] = w;
        fill[v]++;
    }
    return csr;
}

// ---------- Rooting / orders (true postorder, no sort, no revisits) ----------
static void root_tree_iter(
    const CSR& G, int root,
    std::vector<int>& parent, std::vector<double>& parent_w,
    std::vector<int>& depth, std::vector<int>& order_post
){
    const int n = G.n;
    parent.assign(n, -1);
    parent_w.assign(n, 0.0);
    depth.assign(n, 0);
    order_post.clear();
    order_post.reserve(n);

    std::vector<int> it(n, 0);
    std::vector<int> stack;
    stack.reserve(n);

    parent[root] = -1;
    depth[root] = 0;
    stack.push_back(root);

    while (!stack.empty()){
        int u = stack.back();
        int L = static_cast<int>(G.indptr[u]);
        int R = static_cast<int>(G.indptr[u+1]);

        if (it[u] < (R - L)) {
            int k = L + it[u];
            ++it[u];
            int v = G.indices[k];

            if (v == parent[u]) continue;
            if (parent[v] != -1) continue;

            parent[v] = u;
            parent_w[v] = G.weights[k];
            depth[v] = depth[u] + 1;
            stack.push_back(v);
        } else {
            order_post.push_back(u);
            stack.pop_back();
        }
    }
}

static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
    std::vector<uint8_t> contains(parent.size(), 0);
    int u = target;
    while (u != -1) {
        contains[u] = 1;
        u = parent[u];
    }
    return contains;
}

static void compute_sz(
    const std::vector<int>& order_post,
    const std::vector<int>& colors,
    std::vector<long long>& sz0,
    std::vector<long long>& sz1
){
    const int n = static_cast<int>(colors.size());
    sz0.assign(n, 0);
    sz1.assign(n, 0);
    for (int u : order_post) {
        if (colors[u] == 0) sz0[u] += 1;
        else                sz1[u] += 1;
    }
}

static void accumulate_to_parent(
    const std::vector<int>& order_post,
    const std::vector<int>& parent,
    std::vector<long long>& sz0,
    std::vector<long long>& sz1
){
    for (int u : order_post) {
        int p = parent[u];
        if (p != -1){
            sz0[p] += sz0[u];
            sz1[p] += sz1[u];
        }
    }
}

// Compute directed sums along the root->target spine-path for each node on that path.
// out_maxtr[u] (for nodes with same color as root) becomes the "to-target-side" sum at u.
// out_maxtr[root] is set to the total directed sum at root.
static void accumulate_noise_on_branch(
    const CSR& G,
    int root, int target,
    const std::vector<int>& order_post,
    const std::vector<int>& parent,
    const std::vector<int>& colors,
    const std::vector<long long>& sz0,
    const std::vector<long long>& sz1,
    std::vector<double>& dis0,
    std::vector<double>& dis1,
    std::vector<double>& out_maxtr,
    double& total_noise_root
){
    const int n = G.n;
    std::vector<uint8_t> contains = mark_path_to_target(parent, target);

    dis0.assign(n, 0.0);
    dis1.assign(n, 0.0);
    out_maxtr.assign(n, 0.0);

    for (int u : order_post) {
        for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
            int v = G.indices[k];
            if (parent[v] != u) continue;
            if (!contains[v]) continue;

            double w = G.weights[k];
            dis0[u] += dis0[v] + (double)sz0[v] * w;
            dis1[u] += dis1[v] + (double)sz1[v] * w;

            if (colors[u] == colors[root]){
                if (colors[root] == 0) out_maxtr[u] = dis0[v] + (double)sz0[v] * w;
                else                   out_maxtr[u] = dis1[v] + (double)sz1[v] * w;
            }
        }
    }
    total_noise_root = (colors[root] == 0) ? dis0[root] : dis1[root];
}

// Helpers: build numpy arrays (copy) for returning model params
static py::array_t<double> vec_to_np1(const std::vector<double>& v) {
    py::array_t<double> arr({(py::ssize_t)v.size()});
    auto a = arr.mutable_unchecked<1>();
    for (py::ssize_t i=0;i<(py::ssize_t)v.size();++i) a(i) = v[(size_t)i];
    return arr;
}

py::dict fit_core(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                  py::array_t<long long, py::array::c_style | py::array::forcecast> y,
                  double lambda_)
{
    auto t_all_start = Clock::now();

    py::buffer_info bx = X.request();
    py::buffer_info by = y.request();
    if (bx.ndim != 2) throw std::runtime_error("X must be 2D");
    if (by.ndim != 1) throw std::runtime_error("y must be 1D");
    const int64_t N = bx.shape[0];
    const int64_t D = bx.shape[1];
    if (by.shape[0] != N) throw std::runtime_error("X,y length mismatch");
    if (N < 2) throw std::runtime_error("Need at least 2 samples");

    const double* Xptr = static_cast<double*>(bx.ptr);
    const long long* Yptr = static_cast<long long*>(by.ptr);

    // ===== Stage A: means/projections (compute m0, m1, w_unit, t[i], Lleaf[i]) =====
    auto tA0 = Clock::now();

    std::vector<double> m0(D, 0.0), m1(D, 0.0);
    long long c0 = 0, c1 = 0;
    for (int64_t i=0;i<N;i++){
        if (Yptr[i]==0){ c0++; for (int64_t d=0; d<D; ++d) m0[d]+=Xptr[i*D+d]; }
        else            { c1++; for (int64_t d=0; d<D; ++d) m1[d]+=Xptr[i*D+d]; }
    }
    if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 are required.");
    for (int64_t d=0; d<D; ++d){ m0[d]/= (double)c0; m1[d]/= (double)c1; }

    std::vector<double> w(D, 0.0);
    for (int64_t d=0; d<D; ++d) w[d] = m1[d] - m0[d];
    double nw=0.0; for (double v:w) nw += v*v; nw = std::sqrt(nw);
    if (nw==0.0){ w[0]=1.0; for (int64_t d=1; d<D; ++d) w[d]=0.0; }
    else        { for (int64_t d=0; d<D; ++d) w[d]/=nw; }

    std::vector<double> t(N, 0.0), Lleaf(N, 0.0);
    for (int64_t i=0;i<N;i++){
        double ti = 0.0;
        for (int64_t d=0; d<D; ++d) ti += (Xptr[i*D+d] - m0[d])*w[d];
        t[(size_t)i] = ti;

        double res2 = 0.0;
        for (int64_t d=0; d<D; ++d){
            double proj = m0[d] + ti*w[d];
            double r = Xptr[i*D+d] - proj;
            res2 += r*r;
        }
        Lleaf[(size_t)i] = std::sqrt(res2);
    }

    auto tA1 = Clock::now();
    double time_means_proj = secs_since(tA0, tA1);

    // ===== Stage B: sort along spine, compute cumulative spine distances Dsp[pos] =====
    auto tB0 = Clock::now();
    std::vector<int> idx(N);
    for (int i=0;i<(int)N;i++) idx[i]=i;

    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return t[(size_t)a] < t[(size_t)b]; });

    std::vector<double> Dsp(N, 0.0);
    for (int k=1;k<(int)N;k++){
        Dsp[(size_t)k] = Dsp[(size_t)k-1] + std::abs(t[(size_t)idx[(size_t)k]] - t[(size_t)idx[(size_t)k-1]]);
    }
    auto tB1 = Clock::now();
    double time_sort_spine = secs_since(tB0, tB1);

    // ===== Stage C: build augmented tree graph (CSR) =====
    auto tC0 = Clock::now();
    const int TOT = (int)(2*N);
    std::vector<int> colors(TOT, 0);
    for (int i=0;i<(int)N;i++){
        colors[i] = (int)Yptr[i];
        colors[(int)N+i] = (int)Yptr[i];
    }

    std::vector<std::tuple<int,int,double>> edges;
    edges.reserve((size_t)3*N);

    // spoke edges
    for (int i=0;i<(int)N;i++){
        edges.emplace_back(i, (int)N+i, Lleaf[(size_t)i]);
    }
    // spine edges (between consecutive projections in sorted order)
    for (int k=0;k<(int)N-1;k++){
        int i = idx[(size_t)k];
        int j = idx[(size_t)k+1];
        double wsp = std::abs(t[(size_t)j]-t[(size_t)i]);
        edges.emplace_back((int)N+i, (int)N+j, wsp);
    }

    CSR G = build_csr(TOT, edges);
    auto tC1 = Clock::now();
    double time_build_graph = secs_since(tC0, tC1);

    // spine node ids in sorted order
    std::vector<int> spine_ids(N);
    for (int k=0;k<(int)N;k++) spine_ids[(size_t)k] = (int)N + idx[(size_t)k];

    // pos map for spine nodes (optional, but useful)
    std::vector<int> pos_of_node(TOT, -1);
    for (int k=0;k<(int)N;k++) pos_of_node[(size_t)spine_ids[(size_t)k]] = k;

    // ===== Stage D: 4 DP runs to build directional sums for each class =====
    auto tD0 = Clock::now();

    auto compute_side = [&](int root_node, int target_node, std::vector<double>& out_maxtr){
        std::vector<int> parent; std::vector<double> parent_w; std::vector<int> depth; std::vector<int> order_post;
        root_tree_iter(G, root_node, parent, parent_w, depth, order_post);

        std::vector<long long> sz0, sz1;
        compute_sz(order_post, colors, sz0, sz1);
        accumulate_to_parent(order_post, parent, sz0, sz1);

        std::vector<double> dis0, dis1, maxtr;
        double total_noise_root = 0.0;
        accumulate_noise_on_branch(G, root_node, target_node, order_post, parent, colors, sz0, sz1,
                                   dis0, dis1, maxtr, total_noise_root);
        out_maxtr.swap(maxtr);
        out_maxtr[(size_t)root_node] = total_noise_root;
    };

    std::vector<int> X0_sorted; X0_sorted.reserve((size_t)N);
    std::vector<int> X1_sorted; X1_sorted.reserve((size_t)N);
    for (int k=0;k<(int)N;k++){
        int pid = spine_ids[(size_t)k];
        if (colors[(size_t)pid]==0) X0_sorted.push_back(pid);
        else                        X1_sorted.push_back(pid);
    }

    std::vector<double> maxtr0_to_right(TOT, 0.0), maxtr0_to_left(TOT, 0.0),
                        maxtr1_to_right(TOT, 0.0), maxtr1_to_left(TOT, 0.0);

    // Naming clarified:
    // "to_right": sums for direction moving to increasing spine position
    // "to_left" : sums for direction moving to decreasing spine position
    if (X0_sorted.size()>=2){
        compute_side(X0_sorted.front(), X0_sorted.back(),  maxtr0_to_right);
        compute_side(X0_sorted.back(),  X0_sorted.front(), maxtr0_to_left);
    } else if (X0_sorted.size()==1){
        compute_side(X0_sorted.front(), X0_sorted.front(), maxtr0_to_right);
        maxtr0_to_left = maxtr0_to_right;
    }

    if (X1_sorted.size()>=2){
        compute_side(X1_sorted.front(), X1_sorted.back(),  maxtr1_to_right);
        compute_side(X1_sorted.back(),  X1_sorted.front(), maxtr1_to_left);
    } else if (X1_sorted.size()==1){
        compute_side(X1_sorted.front(), X1_sorted.front(), maxtr1_to_right);
        maxtr1_to_left = maxtr1_to_right;
    }

    auto tD1 = Clock::now();
    double time_dp = secs_since(tD0, tD1);

    auto sum_to_right = [&](int node)->double{
        return (colors[(size_t)node]==0) ? maxtr0_to_right[(size_t)node] : maxtr1_to_right[(size_t)node];
    };
    auto sum_to_left = [&](int node)->double{
        return (colors[(size_t)node]==0) ? maxtr0_to_left[(size_t)node] : maxtr1_to_left[(size_t)node];
    };

    auto f_ordered = [&](int u, int v, int pos_u, int pos_v)->double{
        if (pos_u < pos_v) return sum_to_right(u) + sum_to_left(v);
        else               return sum_to_left(u)  + sum_to_right(v);
    };

    auto dist_spine = [&](int pos_u, int pos_v)->double{
        double du = Dsp[(size_t)pos_u];
        double dv = Dsp[(size_t)pos_v];
        return std::abs(dv - du);
    };

    // ===== Stage E: choose scan mode =====
    auto tE0 = Clock::now();
    double best = std::numeric_limits<double>::infinity();
    int best_u=-1, best_v=-1;

    const bool is_unit = (std::abs(lambda_ - 1.0) <= 1e-12);

    if (is_unit) {
        // O(n): scan adjacent edges, but evaluate BOTH directions (u->v and v->u) for completeness
        for (int k=0;k<(int)N-1;k++){
            int u = spine_ids[(size_t)k];
            int v = spine_ids[(size_t)k+1];
            if (colors[(size_t)u]==colors[(size_t)v]) continue;

            double d = Dsp[(size_t)k+1] - Dsp[(size_t)k]; // adjacent distance
            // u->v (pos_u < pos_v)
            double fuv = sum_to_right(u) + sum_to_left(v);
            double Luv = fuv - lambda_ * d;
            if (Luv < best){ best = Luv; best_u = u; best_v = v; }

            // v->u (pos_v > pos_u, direction flips)
            double fvu = sum_to_left(v) + sum_to_right(u);
            double Lvu = fvu - lambda_ * d;
            if (Lvu < best){ best = Lvu; best_u = v; best_v = u; }
        }
    } else {
        // O(n^2): scan ALL unordered pairs (k1<k2) but evaluate BOTH orders => still O(n^2)
        for (int i=0;i<(int)N;i++){
            int u = spine_ids[(size_t)i];
            for (int j=i+1;j<(int)N;j++){
                int v = spine_ids[(size_t)j];
                if (colors[(size_t)u]==colors[(size_t)v]) continue;

                double d = Dsp[(size_t)j] - Dsp[(size_t)i];

                // u->v (i<j)
                double fuv = sum_to_right(u) + sum_to_left(v);
                double Luv = fuv - lambda_ * d;
                if (Luv < best){ best = Luv; best_u = u; best_v = v; }

                // v->u (j>i)
                double fvu = sum_to_left(v) + sum_to_right(u);
                double Lvu = fvu - lambda_ * d;
                if (Lvu < best){ best = Lvu; best_u = v; best_v = u; }
            }
        }
    }

    auto tE1 = Clock::now();
    double time_scan_pairs = secs_since(tE0, tE1);

    if (best_u < 0 || best_v < 0) {
        throw std::runtime_error("No opposite-label pair found (unexpected).");
    }

    // Convert best_u/best_v (tree node id) -> original data index
    int su = (best_u >= (int)N) ? (best_u - (int)N) : best_u;
    int sv = (best_v >= (int)N) ? (best_v - (int)N) : best_v;

    // Build prediction-ready params inside C++:
    // thr = 0.5*(t[su] + t[sv]); left label vs right label determined by t-order.
    double ts = t[(size_t)su];
    double tp = t[(size_t)sv];
    double thr = 0.5 * (ts + tp);

    int y_left, y_right;
    if (ts <= tp) { y_left = (int)Yptr[su]; y_right = (int)Yptr[sv]; }
    else          { y_left = (int)Yptr[sv]; y_right = (int)Yptr[su]; }

    auto t_all_end = Clock::now();
    double time_total = secs_since(t_all_start, t_all_end);

    py::dict out;
    out["support_s"] = su;
    out["support_p"] = sv;
    out["min_val"]   = best;

    // Prediction-ready params (so Python does NOT need to recompute means/w/thr)
    out["m0"] = vec_to_np1(m0);
    out["w"]  = vec_to_np1(w);
    out["thr"] = thr;
    out["y_left"]  = y_left;
    out["y_right"] = y_right;

    // Optional stage times (debug)
    out["time_total"]       = time_total;
    out["time_means_proj"]  = time_means_proj;
    out["time_sort_spine"]  = time_sort_spine;
    out["time_build_graph"] = time_build_graph;
    out["time_dp"]          = time_dp;
    out["time_scan_pairs"]  = time_scan_pairs;

    out["scan_mode"] = is_unit ? "adjacent_O(n)" : "all_pairs_O(n^2)";

    return out;
}

PYBIND11_MODULE(svm_on_tree_cpp, m) {
    m.doc() = "SVM On Tree C++ core (pybind11). Fair-ready: returns model params for E2E prediction.";
    m.def("fit_core", &fit_core, py::arg("X"), py::arg("y"), py::arg("lambda_")=1.0);
    m.attr("BUILD_TAG") = "fair_v1_any_lambda_adj_or_n2";
}
