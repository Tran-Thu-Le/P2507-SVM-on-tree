// SVM On Tree: C++ core with pybind11 bindings.
// Implements the fit_core function for binary classification
// using an augmented tree structure along the spine direction.

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
#include <stdexcept>

namespace py = pybind11;
using Clock = std::chrono::high_resolution_clock;

static inline double secs_since(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double>(b - a).count();
}

// Compressed sparse row representation for undirected trees
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

// Root the tree and compute a true postorder traversal (iterative, no sort)
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

// Mark all nodes on the path from root to target
static std::vector<uint8_t> mark_path_to_target(const std::vector<int>& parent, int target){
    std::vector<uint8_t> contains(parent.size(), 0);
    int u = target;
    while (u != -1) {
        contains[u] = 1;
        u = parent[u];
    }
    return contains;
}

// Bottom-up computation of per-class subtree sizes
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

// Accumulate subtree sizes to parent nodes (bottom-up)
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

// Compute directed distance sums along the root-to-target spine path.
// dis0/dis1 accumulate full subtrees (including leaves).
// out_maxtr[u] stores only the contribution from the child on the path.
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

            double w = G.weights[k];

            // Accumulate full subtree distance sums
            dis0[u] += dis0[v] + (double)sz0[v] * w;
            dis1[u] += dis1[v] + (double)sz1[v] * w;

            // Restrict to the child on the path to target
            if (!contains[v]) continue;

            if (colors[u] == colors[root]){
                if (colors[root] == 0) out_maxtr[u] = dis0[v] + (double)sz0[v] * w;
                else                   out_maxtr[u] = dis1[v] + (double)sz1[v] * w;
            }
        }
    }
    total_noise_root = (colors[root] == 0) ? dis0[root] : dis1[root];
}

// Copy a std::vector<double> into a 1D numpy array
static py::array_t<double> vec_to_np1(const std::vector<double>& v) {
    py::array_t<double> arr((py::ssize_t)v.size());
    auto a = arr.mutable_unchecked<1>();
    for (py::ssize_t i=0;i<(py::ssize_t)v.size();++i) a(i) = v[(size_t)i];
    return arr;
}

// Main fitting function: find the optimal support pair on the augmented tree.
// Returns a dict with support indices, model parameters, and timing information.
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

    // Stage A: compute class means, spine direction, projections, and spoke lengths
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

    // Stage B: sort projections along the spine, compute cumulative spine distances
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

    // Stage C: build the augmented tree graph (2N nodes: N leaves + N spine nodes)
    auto tC0 = Clock::now();
    const int TOT = (int)(2*N);
    std::vector<int> colors(TOT, 0);
    for (int i=0;i<(int)N;i++){
        colors[i] = (int)Yptr[i];
        colors[(int)N+i] = (int)Yptr[i];
    }

    std::vector<std::tuple<int,int,double>> edges;
    edges.reserve((size_t)3*N);

    // Spoke edges connecting each data point to its projection on the spine
    for (int i=0;i<(int)N;i++){
        edges.emplace_back(i, (int)N+i, Lleaf[(size_t)i]);
    }
    // Spine edges between consecutive projections in sorted order
    for (int k=0;k<(int)N-1;k++){
        int i = idx[(size_t)k];
        int j = idx[(size_t)k+1];
        double wsp = std::abs(t[(size_t)j]-t[(size_t)i]);
        edges.emplace_back((int)N+i, (int)N+j, wsp);
    }

    CSR G = build_csr(TOT, edges);
    auto tC1 = Clock::now();
    double time_build_graph = secs_since(tC0, tC1);

    // Spine node indices in sorted order
    std::vector<int> spine_ids(N);
    for (int k=0;k<(int)N;k++) spine_ids[(size_t)k] = (int)N + idx[(size_t)k];

    // Map from node id to spine position
    std::vector<int> pos_of_node(TOT, -1);
    for (int k=0;k<(int)N;k++) pos_of_node[(size_t)spine_ids[(size_t)k]] = k;

    // Stage D: four DP runs to compute directional sums for each class
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

    // Directional sums toward increasing spine position
    if (X0_sorted.size()>=2){
        compute_side(X0_sorted.front(), X0_sorted.back(),  maxtr0_to_right);
        compute_side(X0_sorted.back(),  X0_sorted.front(), maxtr0_to_left);
    } else if (X0_sorted.size()==1){
        compute_side(X0_sorted.front(), X0_sorted.front(), maxtr0_to_right);
        maxtr0_to_left = maxtr0_to_right;
    }

    // Directional sums toward decreasing spine position
    if (X1_sorted.size()>=2){
        compute_side(X1_sorted.front(), X1_sorted.back(),  maxtr1_to_right);
        compute_side(X1_sorted.back(),  X1_sorted.front(), maxtr1_to_left);
    } else if (X1_sorted.size()==1){
        compute_side(X1_sorted.front(), X1_sorted.front(), maxtr1_to_right);
        maxtr1_to_left = maxtr1_to_right;
    }

    auto tD1 = Clock::now();
    double time_dp = secs_since(tD0, tD1);

    auto sum_to_right_spine = [&](int node)->double{
        return (colors[(size_t)node]==0) ? maxtr0_to_right[(size_t)node] : maxtr1_to_right[(size_t)node];
    };
    auto sum_to_left_spine = [&](int node)->double{
        return (colors[(size_t)node]==0) ? maxtr0_to_left[(size_t)node] : maxtr1_to_left[(size_t)node];
    };

    // Stage E: scan candidate pairs to find the optimal support pair
    auto tE0 = Clock::now();
    double best = std::numeric_limits<double>::infinity();
    int best_u=-1, best_v=-1;

    const bool is_unit = (std::abs(lambda_ - 1.0) <= 1e-12);

    if (is_unit) {
        // Lambda = 1: O(n) scan over adjacent spine edges
        for (int k=0;k<(int)N-1;k++){
            int u = spine_ids[(size_t)k];
            int v = spine_ids[(size_t)k+1];
            if (colors[(size_t)u]==colors[(size_t)v]) continue;

            double d = Dsp[(size_t)k+1] - Dsp[(size_t)k];

            double fuv = sum_to_right_spine(u) + sum_to_left_spine(v);
            double Luv = fuv - lambda_ * d;
            if (Luv < best){ best = Luv; best_u = u; best_v = v; }

            double fvu = sum_to_left_spine(v) + sum_to_right_spine(u);
            double Lvu = fvu - lambda_ * d;
            if (Lvu < best){ best = Lvu; best_u = v; best_v = u; }
        }
    } else {
        // Lambda != 1: O(n^2) scan over all node pairs in the augmented tree (2N nodes).
        // Extend spine sums to leaf nodes using spoke length offsets.

        // Spine position for any node (leaf nodes share position with their projection)
        auto pos_node = [&](int node)->int{
            if (node >= (int)N) return pos_of_node[(size_t)node];
            else                return pos_of_node[(size_t)((int)N + node)];
        };
        auto off_node = [&](int node)->double{
            return (node < (int)N) ? Lleaf[(size_t)node] : 0.0;
        };

        // Prefix counts by class along spine positions
        std::vector<int> pref0((int)N+1, 0), pref1((int)N+1, 0);
        for (int k=0;k<(int)N;k++){
            int oi = idx[(size_t)k];
            int c  = (int)Yptr[oi];
            pref0[k+1] = pref0[k] + (c==0);
            pref1[k+1] = pref1[k] + (c==1);
        }
        int total_pos0 = pref0[(int)N];
        int total_pos1 = pref1[(int)N];

        // Count same-label nodes to the right/left (each position has leaf + spine = 2 nodes)
        auto cnt_right_nodes_same_label = [&](int node)->long long{
            int p = pos_node(node);
            int c = colors[(size_t)node];
            int pos_right = (c==0) ? (total_pos0 - pref0[p+1]) : (total_pos1 - pref1[p+1]);
            return 2LL * (long long)pos_right;
        };
        auto cnt_left_nodes_same_label = [&](int node)->long long{
            int p = pos_node(node);
            int c = colors[(size_t)node];
            int pos_left = (c==0) ? pref0[p] : pref1[p];
            return 2LL * (long long)pos_left;
        };

        // Extend directional sums from spine nodes to arbitrary nodes
        auto sum_to_right_any = [&](int node)->double{
            int spine = (node >= (int)N) ? node : ((int)N + node);
            double base = sum_to_right_spine(spine);
            if (node < (int)N) base += (double)cnt_right_nodes_same_label(node) * Lleaf[(size_t)node];
            return base;
        };
        auto sum_to_left_any = [&](int node)->double{
            int spine = (node >= (int)N) ? node : ((int)N + node);
            double base = sum_to_left_spine(spine);
            if (node < (int)N) base += (double)cnt_left_nodes_same_label(node) * Lleaf[(size_t)node];
            return base;
        };

        // Full tree distance between any two nodes
        auto dist_full = [&](int a, int b)->double{
            int pa = pos_node(a), pb = pos_node(b);
            double spine = std::abs(Dsp[(size_t)pa] - Dsp[(size_t)pb]);
            return spine + off_node(a) + off_node(b);
        };

        // Enumerate all opposite-label pairs
        for (int u=0; u<TOT; ++u){
            for (int v=u+1; v<TOT; ++v){
                if (colors[(size_t)u] == colors[(size_t)v]) continue;

                int pu = pos_node(u), pv = pos_node(v);
                if (pu == pv) continue;

                int left  = (pu < pv) ? u : v;
                int right = (pu < pv) ? v : u;

                double f = sum_to_right_any(left) + sum_to_left_any(right);
                double d = dist_full(left, right);
                double L = f - lambda_ * d;

                if (L < best){
                    best = L;
                    best_u = left;
                    best_v = right;
                }
            }
        }
    }

    auto tE1 = Clock::now();
    double time_scan_pairs = secs_since(tE0, tE1);

    if (best_u < 0 || best_v < 0) {
        throw std::runtime_error("No opposite-label pair found (unexpected).");
    }

    // Map tree node id back to original data index in [0, N-1]
    int su = (best_u >= (int)N) ? (best_u - (int)N) : best_u;
    int sv = (best_v >= (int)N) ? (best_v - (int)N) : best_v;

    // Compute threshold as the midpoint of support projections
    double ts = t[(size_t)su];
    double tp = t[(size_t)sv];
    double thr = 0.5 * (ts + tp);

    int y_left, y_right;
    if (ts <= tp) { y_left = (int)Yptr[su]; y_right = (int)Yptr[sv]; }
    else          { y_left = (int)Yptr[sv]; y_right = (int)Yptr[su]; }

    auto t_all_end = Clock::now();
    double time_total = secs_since(t_all_start, t_all_end);

    // Assemble output dictionary
    py::dict out;
    out["support_s"] = su;
    out["support_p"] = sv;
    out["min_val"]   = best;

    // Model parameters for prediction
    out["m0"] = vec_to_np1(m0);
    out["w"]  = vec_to_np1(w);
    out["thr"] = thr;
    out["y_left"]  = y_left;
    out["y_right"] = y_right;

    // Stage timing
    out["time_total"]       = time_total;
    out["time_means_proj"]  = time_means_proj;
    out["time_sort_spine"]  = time_sort_spine;
    out["time_build_graph"] = time_build_graph;
    out["time_dp"]          = time_dp;
    out["time_scan_pairs"]  = time_scan_pairs;

    out["scan_mode"] = is_unit ? "adjacent_O(n)" : "all_nodes_O(n^2)";

    return out;
}

PYBIND11_MODULE(svm_on_tree_cpp, m) {
    m.doc() = "SVM On Tree C++ core with pybind11 bindings.";
    m.def("fit_core", &fit_core, py::arg("X"), py::arg("y"), py::arg("lambda_")=1.0);
    m.attr("BUILD_TAG") = "v2_any_lambda";
}
