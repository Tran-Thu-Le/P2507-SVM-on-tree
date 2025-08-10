#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

namespace py = pybind11;

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

// ---------- Rooting / orders ----------
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
    // post-order: sort by depth desc
    order_post = order_pre;
    std::sort(order_post.begin(), order_post.end(),
              [&](int a, int b){ return depth[a] > depth[b]; });
}

// ---------- Helpers ----------
static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
    std::vector<uint8_t> contains(parent.size(), 0);
    int u = target;
    while (u != -1) {
        contains[u] = 1;
        u = parent[u];
    }
    return contains;
}

// ---------- sz, dis accumulations on the branch ----------
static void compute_sz(
    const CSR& G,
    const std::vector<int>& order_post,
    const std::vector<int>& colors,
    std::vector<long long>& sz1,
    std::vector<long long>& sz2
){
    const int n = G.n;
    sz1.assign(n, 0);
    sz2.assign(n, 0);
    // count self
    for (int u : order_post) {
        if (colors[u] == 0) sz1[u] += 1;
        else                sz2[u] += 1;
    }
    // push up to parent
    // We need parent, but we can reconstruct via a separate pass (caller has parent)
}

// accumulate sz to parent using parent array
static void accumulate_to_parent(
    const std::vector<int>& order_post,
    const std::vector<int>& parent,
    std::vector<long long>& sz1,
    std::vector<long long>& sz2
){
    for (int u : order_post) {
        int p = parent[u];
        if (p != -1){
            sz1[p] += sz1[u];
            sz2[p] += sz2[u];
        }
    }
}

static void accumulate_noise_on_branch(
    const CSR& G,
    int root, int target,
    const std::vector<int>& order_post,
    const std::vector<int>& parent,
    const std::vector<double>& parent_w,
    const std::vector<int>& colors,
    const std::vector<long long>& sz1,
    const std::vector<long long>& sz2,
    std::vector<double>& dis_s,
    std::vector<double>& dis_p,
    std::vector<double>& maxtr,
    double& total_noise_root
){
    const int n = G.n;
    std::vector<uint8_t> contains = mark_path_to_target(parent, target);
    dis_s.assign(n, 0.0);
    dis_p.assign(n, 0.0);
    maxtr.assign(n, 0.0);

    // bottom-up: only children on the path root->target contribute
    for (int u : order_post) {
        // iterate children v (neighbors with parent[v]==u)
        for (long long k = G.indptr[u]; k < G.indptr[u+1]; ++k) {
            int v = G.indices[k];
            if (parent[v] != u) continue;
            if (!contains[v]) continue;
            double w = G.weights[k];
            dis_s[u] += dis_s[v] + (double)sz1[v] * w;
            dis_p[u] += dis_p[v] + (double)sz2[v] * w;
            if (colors[u] == colors[root]){
                if (colors[root] == 0) maxtr[u] = dis_s[v] + (double)sz1[v] * w;
                else                   maxtr[u] = dis_p[v] + (double)sz2[v] * w;
            }
        }
    }
    total_noise_root = (colors[root] == 0) ? dis_s[root] : dis_p[root];
}

// ---------- Main core: compute support pair ----------
py::dict fit_core(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                  py::array_t<long long, py::array::c_style | py::array::forcecast> y,
                  double lambda_)
{
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

    // means m0, m1
    std::vector<double> m0(D, 0.0), m1(D, 0.0);
    long long c0 = 0, c1 = 0;
    for (int64_t i=0;i<N;i++){
        if (Yptr[i]==0){ c0++; for (int64_t d=0; d<D; ++d) m0[d]+=Xptr[i*D+d]; }
        else            { c1++; for (int64_t d=0; d<D; ++d) m1[d]+=Xptr[i*D+d]; }
    }
    if (c0==0 || c1==0) throw std::runtime_error("Both classes 0 and 1 are required.");
    for (int64_t d=0; d<D; ++d){ m0[d]/= (double)c0; m1[d]/= (double)c1; }

    // w = m1 - m0, normalize
    std::vector<double> w(D, 0.0);
    for (int64_t d=0; d<D; ++d) w[d] = m1[d] - m0[d];
    double nw=0.0; for (double v:w) nw += v*v; nw = std::sqrt(nw);
    if (nw==0.0){ w[0]=1.0; for (int64_t d=1; d<D; ++d) w[d]=0.0; }
    else        { for (int64_t d=0; d<D; ++d) w[d]/=nw; }

    // t_all, projections, residual norms (leaf->proj weight)
    std::vector<double> t(N, 0.0), Lleaf(N, 0.0);
    std::vector<double> Xproj(N*D, 0.0);
    for (int64_t i=0;i<N;i++){
        double ti = 0.0;
        for (int64_t d=0; d<D; ++d) ti += (Xptr[i*D+d] - m0[d])*w[d];
        t[i] = ti;
        for (int64_t d=0; d<D; ++d) Xproj[i*D+d] = m0[d] + ti*w[d];
        double res2 = 0.0;
        for (int64_t d=0; d<D; ++d){
            double r = Xptr[i*D+d] - Xproj[i*D+d];
            res2 += r*r;
        }
        Lleaf[i] = std::sqrt(res2);
    }

    // sort by t
    std::vector<int> idx(N); for (int i=0;i<N;i++) idx[i]=i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return t[a] < t[b]; });

    // spine coordinates D[k]
    std::vector<double> Dsp(N, 0.0);
    for (int k=1;k<N;k++){
        Dsp[k] = Dsp[k-1] + std::abs(t[idx[k]] - t[idx[k-1]]);
    }

    // Build tree: nodes 0..N-1 (leaves), N..2N-1 (projections)
    const int TOT = (int)(2*N);
    std::vector<int> colors(TOT, 0);
    for (int i=0;i<N;i++){
        colors[i] = (int)Yptr[i];
        colors[N+i] = (int)Yptr[i];
    }
    std::vector<std::tuple<int,int,double>> edges;
    edges.reserve(3*N);
    for (int i=0;i<N;i++){
        edges.emplace_back(i, N+i, Lleaf[i]);
    }
    for (int k=0;k<N-1;k++){
        int i = idx[k], j = idx[k+1];
        double wsp = std::abs(t[j]-t[i]); // == Dsp[k+1]-Dsp[k]
        edges.emplace_back(N+i, N+j, wsp);
    }
    CSR G = build_csr(TOT, edges);

    // Prepare boundary sets along spine
    std::vector<int> spine_ids(N);
    for (int k=0;k<N;k++) spine_ids[k] = N + idx[k];

    auto compute_side = [&](int root_node, int target_node, std::vector<double>& out_maxtr){
        std::vector<int> parent; std::vector<double> parent_w; std::vector<int> depth; std::vector<int> order_post;
        root_tree_iter(G, root_node, parent, parent_w, depth, order_post);

        std::vector<long long> sz1, sz2;
        compute_sz(G, order_post, colors, sz1, sz2);
        accumulate_to_parent(order_post, parent, sz1, sz2);

        std::vector<double> dis_s, dis_p, maxtr;
        double total_noise_root = 0.0;
        accumulate_noise_on_branch(G, root_node, target_node, order_post, parent, parent_w,
                                   colors, sz1, sz2, dis_s, dis_p, maxtr, total_noise_root);
        out_maxtr = maxtr;
        // store total noise at root in out_maxtr[root]
        out_maxtr[root_node] = total_noise_root;
    };

    // Build class-specific boundary lists
    std::vector<int> X0_sorted, X1_sorted;
    X0_sorted.reserve(N); X1_sorted.reserve(N);
    for (int k=0;k<N;k++){
        int pid = spine_ids[k];
        if (colors[pid]==0) X0_sorted.push_back(pid);
        else                X1_sorted.push_back(pid);
    }

    std::vector<double> maxtr_X0_left(TOT, 0.0), maxtr_X0_right(TOT, 0.0),
                        maxtr_X1_left(TOT, 0.0), maxtr_X1_right(TOT, 0.0);

    if (X0_sorted.size()>=2){
        compute_side(X0_sorted.front(), X0_sorted.back(),  maxtr_X0_left);
        compute_side(X0_sorted.back(),  X0_sorted.front(), maxtr_X0_right);
    } else if (X0_sorted.size()==1){
        compute_side(X0_sorted.front(), X0_sorted.front(), maxtr_X0_left);
        maxtr_X0_right = maxtr_X0_left;
    }

    if (X1_sorted.size()>=2){
        compute_side(X1_sorted.front(), X1_sorted.back(),  maxtr_X1_left);
        compute_side(X1_sorted.back(),  X1_sorted.front(), maxtr_X1_right);
    } else if (X1_sorted.size()==1){
        compute_side(X1_sorted.front(), X1_sorted.front(), maxtr_X1_left);
        maxtr_X1_right = maxtr_X1_left;
    }

    // scan adjacent pairs on spine of different color
    double best = std::numeric_limits<double>::infinity();
    int best_s=-1, best_p=-1;
    for (int k=0;k<N-1;k++){
        int s = spine_ids[k];
        int p = spine_ids[k+1];
        if (colors[s]==colors[p]) continue;
        double dist_sp = Dsp[k+1]-Dsp[k]; // edge weight between s and p
        double ans = 0.0;
        if (colors[s]==0){
            ans = maxtr_X0_left[s] + maxtr_X1_right[p] - lambda_ * dist_sp;
        } else {
            ans = maxtr_X1_left[s] + maxtr_X0_right[p] - lambda_ * dist_sp;
        }
        if (ans < best){
            best = ans; best_s = s; best_p = p;
        }
    }

    // map projection node back to original sample index
    // projection node "N+i" corresponds to original index "i"
    int support_s_idx = (best_s>=N) ? (best_s - N) : best_s;
    int support_p_idx = (best_p>=N) ? (best_p - N) : best_p;

    py::dict out;
    out["support_s"] = support_s_idx;  // index trong X
    out["support_p"] = support_p_idx;  // index trong X
    out["min_val"]   = best;
    return out;
}

PYBIND11_MODULE(svm_on_tree_cpp, m) {
    m.doc() = "C++ core for SVM On Tree (pybind11)";
    m.def("fit_core", &fit_core, py::arg("X"), py::arg("y"), py::arg("lamda")=1.0,
          R"pbdoc(
              Core routine for SVM On Tree:
              Inputs:
                X: (N, d) float64
                y: (N,) int64 with labels 0/1
                lamda: float
              Returns dict: {support_s, support_p, min_val}
          )pbdoc");
}
