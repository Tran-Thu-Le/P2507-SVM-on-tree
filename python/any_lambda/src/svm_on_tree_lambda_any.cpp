#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

//====================== CSR (undirected tree) ======================//
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
    CSR csr; csr.n = n;
    csr.indptr.assign(n + 1, 0);
    for (int i=0;i<n;i++) csr.indptr[i+1] = csr.indptr[i] + deg[i];
    csr.indices.assign(csr.indptr.back(), 0);
    csr.weights.assign(csr.indptr.back(), 0.0);

    std::vector<int> fill(n, 0);
    for (auto &e : edges) {
        int u, v; double w;
        std::tie(u, v, w) = e;
        long long pu = csr.indptr[u] + fill[u];
        csr.indices[pu] = v; csr.weights[pu] = w; fill[u]++;
        long long pv = csr.indptr[v] + fill[v];
        csr.indices[pv] = u; csr.weights[pv] = w; fill[v]++;
    }
    return csr;
}

//====================== Rooting & orders ======================//
static void root_tree_iter(
    const CSR& G, int root,
    std::vector<int>& parent, std::vector<double>& parent_w,
    std::vector<int>& depth, std::vector<int>& order_post
){
    const int n = G.n;
    parent.assign(n, -1);
    parent_w.assign(n, 0.0);
    depth.assign(n, 0);
    std::vector<int> stack; stack.reserve(n);
    std::vector<int> order_pre; order_pre.reserve(n);

    stack.push_back(root);
    parent[root] = -1;
    while (!stack.empty()) {
        int u = stack.back(); stack.pop_back();
        order_pre.push_back(u);
        for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
            int v = G.indices[k];
            if (v == parent[u]) continue;
            parent[v] = u;
            parent_w[v] = G.weights[k];
            depth[v] = depth[u] + 1;
            stack.push_back(v);
        }
    }
    order_post = order_pre;
    std::sort(order_post.begin(), order_post.end(),
              [&](int a, int b){ return depth[a] > depth[b]; });
}

static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
    std::vector<uint8_t> contains(parent.size(), 0);
    for (int u = target; u != -1; u = parent[u]) contains[u] = 1;
    return contains;
}

//====================== sz & noise accum on root->target path ======================//
static void accumulate_noise_on_branch(
    const CSR& G,
    int root, int target,
    const std::vector<int>& order_post,
    const std::vector<int>& parent,
    const std::vector<int>& colors,
    std::vector<double>& dis0,
    std::vector<double>& dis1,
    std::vector<double>& maxtr, // contribution of the child-on-path subtree at u
    double& total_noise_root    // tol at root (class-consistent)
){
    const int n = G.n;
    std::vector<uint8_t> on_path = mark_path_to_target(parent, target);
    dis0.assign(n, 0.0); dis1.assign(n, 0.0);
    maxtr.assign(n, 0.0);

    // compute subtree sizes (counts) by a bottom-up pass
    std::vector<long long> sz0(n,0), sz1(n,0);
    for (int u : order_post) {
        if (colors[u] == 0) sz0[u] += 1; else sz1[u] += 1;
    }
    for (int u : order_post) {
        int p = parent[u];
        if (p != -1){ sz0[p] += sz0[u]; sz1[p] += sz1[u]; }
    }

    for (int u : order_post) {
        for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
            int v = G.indices[k];
            if (parent[v] != u) continue;    // child
            if (!on_path[v]) continue;       // only child on root->target path contributes
            double w = G.weights[k];
            dis0[u] += dis0[v] + (double)sz0[v] * w;
            dis1[u] += dis1[v] + (double)sz1[v] * w;

            if (colors[u] == colors[root]) {
                if (colors[root] == 0) maxtr[u] = dis0[v] + (double)sz0[v]*w;
                else                   maxtr[u] = dis1[v] + (double)sz1[v]*w;
            }
        }
    }
    total_noise_root = (colors[root]==0) ? dis0[root] : dis1[root];
}

//====================== Main: arbitrary lambda ======================//
py::dict fit_core_lambda_any(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<long long, py::array::c_style | py::array::forcecast> y,
    double lambda_
){
    auto bx = X.request(); auto by = y.request();
    if (bx.ndim != 2) throw std::runtime_error("X must be 2D");
    if (by.ndim != 1) throw std::runtime_error("y must be 1D");
    const int64_t N = bx.shape[0]; const int64_t D = bx.shape[1];
    if (by.shape[0] != N) throw std::runtime_error("X,y length mismatch");
    if (N < 2) throw std::runtime_error("Need at least 2 samples");
    const double* Xptr = static_cast<double*>(bx.ptr);
    const long long* Yptr = static_cast<long long*>(by.ptr);

    // means m0/m1, direction w
    std::vector<double> m0(D,0.0), m1(D,0.0);
    long long c0=0,c1=0;
    for (int64_t i=0;i<N;i++){
        if (Yptr[i]==0){ c0++; for (int64_t d=0; d<D; ++d) m0[d]+=Xptr[i*D+d]; }
        else            { c1++; for (int64_t d=0; d<D; ++d) m1[d]+=Xptr[i*D+d]; }
    }
    if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 required.");
    for (int64_t d=0; d<D; ++d){ m0[d]/=(double)c0; m1[d]/=(double)c1; }
    std::vector<double> w(D,0.0);
    for (int64_t d=0; d<D; ++d) w[d]=m1[d]-m0[d];
    double nw=0.0; for(double v:w) nw+=v*v; nw=std::sqrt(nw);
    if (nw==0.0){ w[0]=1.0; for (int64_t d=1; d<D; ++d) w[d]=0.0; } else
        for (int64_t d=0; d<D; ++d) w[d]/=nw;

    // projections and spoke lengths
    std::vector<double> t(N,0.0), Lspoke(N,0.0);
    std::vector<double> Xproj(N*D,0.0);
    for (int64_t i=0;i<N;i++){
        double ti=0.0;
        for (int64_t d=0; d<D; ++d) ti += (Xptr[i*D+d]-m0[d])*w[d];
        t[i]=ti;
        for (int64_t d=0; d<D; ++d) Xproj[i*D+d]=m0[d]+ti*w[d];
        double res2=0.0;
        for (int64_t d=0; d<D; ++d){
            double r = Xptr[i*D+d]-Xproj[i*D+d];
            res2 += r*r;
        }
        Lspoke[i]=std::sqrt(res2);
    }

    // sort by t (spine order)
    std::vector<int> idx(N); for(int i=0;i<N;i++) idx[i]=i;
    std::sort(idx.begin(), idx.end(), [&](int a,int b){ return t[a]<t[b]; });

    // spine cumulative coordinate S (0-based over N projections)
    std::vector<double> S(N,0.0);
    for (int k=1;k<N;k++){
        S[k]=S[k-1]+std::abs(t[idx[k]]-t[idx[k-1]]);
    }

    // Build tree nodes: [0..N-1]=original, [N..2N-1]=projection
    const int TOT = (int)(2*N);
    std::vector<int> colors(TOT,0);
    for (int i=0;i<N;i++){ colors[i]=(int)Yptr[i]; colors[N+i]=(int)Yptr[i]; }

    // edges: spoke + spine
    std::vector<std::tuple<int,int,double>> edges; edges.reserve(3*N);
    for (int i=0;i<N;i++) edges.emplace_back(i, N+i, Lspoke[i]);
    for (int k=0;k<N-1;k++){
        int a = idx[k], b = idx[k+1];
        edges.emplace_back(N+a, N+b, std::abs(t[b]-t[a]));
    }
    CSR G = build_csr(TOT, edges);

    // mapping: spine order -> node id of projection
    std::vector<int> spine_ids(N);
    for (int k=0;k<N;k++) spine_ids[k]=N+idx[k];

    //--- DP to get noise arrays along spine for each class and direction ---//
    auto compute_side = [&](int root_node, int target_node, std::vector<double>& out_maxtr, double& tol_root){
        std::vector<int> parent, depth, order_post;
        std::vector<double> parent_w;
        root_tree_iter(G, root_node, parent, parent_w, depth, order_post);

        std::vector<double> d0,d1,maxtr;
        double total_noise_root = 0.0;
        accumulate_noise_on_branch(G, root_node, target_node, order_post, parent, colors,
                                   d0,d1,maxtr,total_noise_root);
        out_maxtr = maxtr;
        tol_root = total_noise_root;
    };

    // tách spine theo lớp
    std::vector<int> X0_sorted, X1_sorted; X0_sorted.reserve(N); X1_sorted.reserve(N);
    for (int k=0;k<N;k++){
        int pid = spine_ids[k];
        if (colors[pid]==0) X0_sorted.push_back(pid); else X1_sorted.push_back(pid);
    }

    std::vector<double> max0L(TOT,0.0), max0R(TOT,0.0), max1L(TOT,0.0), max1R(TOT,0.0);
    double dumpTol=0.0;
    if (!X0_sorted.empty()){
        int L = X0_sorted.front(), R = X0_sorted.back();
        compute_side(L, R, max0L, dumpTol); // “đi về phải”
        compute_side(R, L, max0R, dumpTol); // “đi về trái”
    }
    if (!X1_sorted.empty()){
        int L = X1_sorted.front(), R = X1_sorted.back();
        compute_side(L, R, max1L, dumpTol);
        compute_side(R, L, max1R, dumpTol);
    }

    // vị trí projection trên spine: pos[node] in [0..N-1]
    std::vector<int> pos(TOT, -1);
    for (int k=0;k<N;k++){
        int pid = spine_ids[k];
        pos[pid]=k;
    }
    for (int i=0;i<N;i++){
        pos[i] = pos[N+i]; // original chia sẻ pos với projection
    }

    // prefix-count số projections mỗi lớp trên spine
    std::vector<int> cls_at(N,0);
    for (int k=0;k<N;k++){
        int pid = spine_ids[k]; cls_at[k]=colors[pid];
    }
    std::vector<int> pref0(N+1,0), pref1(N+1,0);
    for (int k=0;k<N;k++){
        pref0[k+1]=pref0[k]+(cls_at[k]==0);
        pref1[k+1]=pref1[k]+(cls_at[k]==1);
    }
    int total0=pref0[N], total1=pref1[N];

    auto left_same = [&](int c, int k)->int{
        return (c==0)? pref0[k] : pref1[k];
    };
    auto right_same = [&](int c, int k)->int{
        return (c==0)? (total0 - pref0[k+1]) : (total1 - pref1[k+1]);
    };

    auto noise_proj = [&](int cls, bool dir_right, int u_proj)->double{
        if (cls==0) return dir_right ? max0L[u_proj] : max0R[u_proj];
        else        return dir_right ? max1L[u_proj] : max1R[u_proj];
    };

    // f_side(u,v): nhiễu phía cùng nhãn với u khi root=u và đi theo nhánh tới v
    auto f_side = [&](int u, int v)->double{
        int cu = colors[u];
        int iu = pos[u], iv = pos[v];
        bool dir_right = (iv >= iu);
        int u_proj = (u < (int)N) ? (N+u) : u;
        double f = noise_proj(cu, dir_right, u_proj);

        // Nếu u là original node, các lá cùng nhãn phía xét (kể cả u) còn đi thêm spoke(u).
        // Ở bản này, spoke được cộng vào d(u,v) để tránh đếm đôi trong f.
        (void)left_same; (void)right_same; // giữ cho compilers không cảnh báo nếu không dùng
        return f;
    };

    // khoảng cách spine(u', v') + spoke(u) + spoke(v)
    auto dist_uv = [&](int u, int v)->double{
        int iu = pos[u], iv = pos[v];
        double spine = std::abs(S[iv] - S[iu]);
        double su = (u < (int)N) ? Lspoke[u] : 0.0;
        double sv = (v < (int)N) ? Lspoke[v] : 0.0;
        return spine + su + sv;
    };

    double best = std::numeric_limits<double>::infinity();
    int best_u=-1, best_v=-1; double best_d=0.0;

    for (int u=0; u<(int)TOT; ++u){
        int cu = colors[u]; if (cu!=0 && cu!=1) continue;
        for (int v=0; v<(int)TOT; ++v){
            if (u==v) continue;
            int cv = colors[v]; if (cv!=0 && cv!=1) continue;
            if (cu == cv) continue;

            double f = f_side(u,v) + f_side(v,u);
            double d = dist_uv(u,v);
            double L = f - lambda_ * d;
            if (L < best){ best=L; best_u=u; best_v=v; best_d=d; }
        }
    }

    auto to_orig = [&](int node)->int{ return (node < (int)N)? node : (node - (int)N); };
    bool s_is_proj = (best_u >= (int)N);
    bool p_is_proj = (best_v >= (int)N);

    py::dict out;
    out["s_node"]   = best_u;
    out["p_node"]   = best_v;
    out["s_is_proj"]= s_is_proj;
    out["p_is_proj"]= p_is_proj;
    out["s_orig"]   = to_orig(best_u);
    out["p_orig"]   = to_orig(best_v);
    out["min_val"]  = best;
    out["dist"]     = best_d;
    out["lambda"]   = lambda_;
    return out;
}

PYBIND11_MODULE(svm_on_tree_cpp, m){
    m.doc() = "C++ core for SVM-on-Tree (arbitrary lambda) — all DP/DFS inside C++";
    m.def("fit_core_lambda_any", &fit_core_lambda_any,
          py::arg("X"), py::arg("y"), py::arg("lamda")=1.0,
          R"pbdoc(
            Compute optimal (s,p) for arbitrary lambda.
            Inputs:
              X: (N,d) float64, y: (N,) int64 in {0,1}, lamda: float
            Returns dict:
              {s_node, p_node, s_is_proj, p_is_proj, s_orig, p_orig, min_val, dist, lambda}
          )pbdoc");
}
