"""Problem 41 - Clustering districts by deprivation profile (National MPI, NITI Aayog 2023).
Runs every analysis in the paper; writes results/results.json and results/*.png"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
SEED = 42; R = {}
raw = pd.read_csv("niti_mpi_districts_extracted.csv")
F4 = ["Headcount_NFHS4", "Intensity_NFHS4", "Headcount_NFHS5", "Intensity_NFHS5"]
R["n_extracted"] = len(raw); R["n_states"] = int(raw.State.nunique())
R["missing"] = {c: int(raw[c].isna().sum()) for c in raw.columns[2:]}
df = raw.dropna(subset=F4).reset_index(drop=True); n = len(df); R["n"] = n
R["n_excluded_nfhs4"] = int(raw[["Headcount_NFHS4", "Intensity_NFHS4"]].isna().any(axis=1).sum())
df["dH"] = df.Headcount_NFHS5 - df.Headcount_NFHS4
desc = df[["Headcount_NFHS4", "Intensity_NFHS4", "MPI_NFHS4", "Headcount_NFHS5", "Intensity_NFHS5", "MPI_NFHS5", "dH"]].agg(["mean", "std", "min", "median", "max"]).T
R["desc"] = desc.round(3).to_dict("index")
X = StandardScaler().fit_transform(df[F4]); R["improved_share"] = float((df.dH < 0).mean()); R["n_worse"] = int((df.dH > 0).sum())

# correlation heatmap
C = df[["Headcount_NFHS4", "Intensity_NFHS4", "MPI_NFHS4", "Headcount_NFHS5", "Intensity_NFHS5", "MPI_NFHS5"]].corr()
R["corr"] = C.round(3).to_dict()
lab = ["H4", "A4", "MPI4", "H5", "A5", "MPI5"]
fig, ax = plt.subplots(figsize=(3.4, 2.9)); im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(lab); ax.set_yticks(range(6)); ax.set_yticklabels(lab)
for i in range(6):
    for j in range(6): ax.text(j, i, f"{C.values[i,j]:.2f}", ha="center", va="center", fontsize=6.5)
plt.colorbar(im, shrink=.8); plt.tight_layout(); plt.savefig("results/fig_corr.png", dpi=220); plt.close()

# distributions
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.2))
ax[0].hist(df.Headcount_NFHS4, bins=30, color="#7a7a9d"); ax[0].set_title("Headcount ratio, NFHS-4 (%)"); ax[0].set_ylabel("Districts")
ax[1].hist(df.Headcount_NFHS5, bins=30, color="#b04a5a"); ax[1].set_title("Headcount ratio, NFHS-5 (%)")
plt.tight_layout(); plt.savefig("results/fig_hist.png", dpi=220); plt.close()

# PCA
pca = PCA().fit(X); ev = pca.explained_variance_ratio_; R["pca_ev"] = ev.round(4).tolist()
R["pca_load"] = {f"PC{i+1}": dict(zip(F4, pca.components_[i].round(3))) for i in range(2)}
Z = PCA(2).fit_transform(X)
fig, ax = plt.subplots(figsize=(3.4, 2.2)); ax.bar(range(1, 5), ev * 100, color="#4c6a92"); ax.plot(range(1, 5), np.cumsum(ev) * 100, "o-", c="#c8553d")
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)"); ax.set_xticks(range(1, 5)); plt.tight_layout(); plt.savefig("results/fig_scree.png", dpi=220); plt.close()

# k selection
rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=20, random_state=SEED).fit(X)
    rows.append(dict(k=k, SSE=km.inertia_, Sil=silhouette_score(X, km.labels_), DB=davies_bouldin_score(X, km.labels_), CH=calinski_harabasz_score(X, km.labels_)))
sel = pd.DataFrame(rows); R["sel"] = sel.round(4).to_dict("records")
fig, ax = plt.subplots(2, 2, figsize=(3.4, 3.2)); ax = ax.ravel()
ax[0].plot(sel.k, sel.SSE, "o-"); ax[0].set_title("Elbow (SSE)", fontsize=7); ax[1].plot(sel.k, sel.Sil, "s-", c="C1"); ax[1].set_title("Silhouette", fontsize=7)
ax[2].plot(sel.k, sel.DB, "^-", c="C2"); ax[2].set_title("Davies-Bouldin", fontsize=7); ax[3].plot(sel.k, sel.CH, "d-", c="C3"); ax[3].set_title("Calinski-Harabasz", fontsize=7)
for a_ in ax: a_.set_xlabel("k", fontsize=7); a_.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig("results/fig_k.png", dpi=220); plt.close()

# stability by k (bootstrap ARI)
rng = np.random.default_rng(SEED); stab = {}
for k in range(2, 6):
    ref = KMeans(k, n_init=20, random_state=SEED).fit(X); a = []
    for b in range(100):
        idx = rng.integers(0, n, n); m = KMeans(k, n_init=5, random_state=b).fit(X[idx]); a.append(adjusted_rand_score(ref.labels_, m.predict(X)))
    stab[k] = (float(np.mean(a)), float(np.std(a)))
R["stab"] = stab

# final model K=3, relabel by H5 descending
K = 3; km = KMeans(K, n_init=50, random_state=SEED).fit(X)
order = df.assign(l=km.labels_).groupby("l").Headcount_NFHS5.mean().sort_values(ascending=False).index.tolist()
mp = {o: i for i, o in enumerate(order)}; lab_km = np.array([mp[l] for l in km.labels_]); df["Cluster"] = lab_km
names = ["High", "Medium", "Low"]; df["Tier"] = df.Cluster.map(dict(enumerate(names)))
# methods
def metr(l): return dict(Sil=silhouette_score(X, l), DB=davies_bouldin_score(X, l), CH=calinski_harabasz_score(X, l), ARI=adjusted_rand_score(lab_km, l))
M = {"k-means": metr(lab_km)}
M["Ward"] = metr(AgglomerativeClustering(K, linkage="ward").fit_predict(X))
M["Average linkage"] = metr(AgglomerativeClustering(K, linkage="average").fit_predict(X))
M["Complete linkage"] = metr(AgglomerativeClustering(K, linkage="complete").fit_predict(X))
M["Gaussian mixture"] = metr(GaussianMixture(K, covariance_type="full", n_init=10, random_state=SEED).fit(X).predict(X))
Xc = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
M["k-means (cosine)"] = dict(Sil=silhouette_score(Xc, KMeans(K, n_init=20, random_state=SEED).fit_predict(Xc)), DB=np.nan, CH=np.nan, ARI=adjusted_rand_score(lab_km, KMeans(K, n_init=20, random_state=SEED).fit_predict(Xc)))
R["methods"] = {k: {m: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)) for m, v in d.items()} for k, d in M.items()}
R["cluster_sizes"] = df.Cluster.value_counts().sort_index().tolist()

# profiles
prof = df.groupby("Cluster").agg(n=("District", "size"), H4=("Headcount_NFHS4", "mean"), H5=("Headcount_NFHS5", "mean"), H5sd=("Headcount_NFHS5", "std"), A4=("Intensity_NFHS4", "mean"), A5=("Intensity_NFHS5", "mean"), MPI5=("MPI_NFHS5", "mean"), dH=("dH", "mean"))
prof["rel"] = (prof.H5 - prof.H4) / prof.H4 * 100; R["prof"] = prof.round(3).to_dict("index")
R["sil_by_cluster"] = {}
from sklearn.metrics import silhouette_samples
ss = silhouette_samples(X, lab_km); R["sil_by_cluster"] = {int(c): float(ss[lab_km == c].mean()) for c in range(K)}; R["sil_neg_share"] = float((ss < 0).mean())

# plots: PCA scatter, H4 vs H5, dendrogram, boxplots
col = ["#b2182b", "#e0a030", "#2166ac"]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
for c in range(K): ax.scatter(Z[lab_km == c, 0], Z[lab_km == c, 1], s=7, c=col[c], label=f"{names[c]} (n={int((lab_km==c).sum())})")
ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})"); ax.legend(fontsize=6, frameon=False); plt.tight_layout(); plt.savefig("results/fig_pca.png", dpi=220); plt.close()
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for c in range(K): ax.scatter(df.Headcount_NFHS4[lab_km == c], df.Headcount_NFHS5[lab_km == c], s=7, c=col[c], label=names[c])
mx = max(df.Headcount_NFHS4.max(), df.Headcount_NFHS5.max()); ax.plot([0, mx], [0, mx], "k--", lw=.7); ax.text(mx * .62, mx * .72, "no change", fontsize=6, rotation=38)
ax.set_xlabel("Headcount ratio NFHS-4 (%)"); ax.set_ylabel("Headcount ratio NFHS-5 (%)"); ax.legend(fontsize=6, frameon=False); plt.tight_layout(); plt.savefig("results/fig_h4h5.png", dpi=220); plt.close()
Lk = linkage(X, "ward"); fig, ax = plt.subplots(figsize=(3.4, 2.2)); dendrogram(Lk, truncate_mode="lastp", p=24, no_labels=True, ax=ax, color_threshold=Lk[-2, 2] * 0.999, above_threshold_color="gray")
ax.set_ylabel("Ward distance"); ax.set_xlabel("Merged groups (truncated)"); plt.tight_layout(); plt.savefig("results/fig_dendro.png", dpi=220); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.2))
ax[0].boxplot([df.Headcount_NFHS5[lab_km == c] for c in range(K)], tick_labels=["H", "M", "L"]); ax[0].set_title("Headcount NFHS-5 (%)", fontsize=7)
ax[1].boxplot([df.Intensity_NFHS5[lab_km == c] for c in range(K)], tick_labels=["H", "M", "L"]); ax[1].set_title("Intensity NFHS-5 (%)", fontsize=7)
plt.tight_layout(); plt.savefig("results/fig_box.png", dpi=220); plt.close()

# convergence
r = float(np.corrcoef(df.Headcount_NFHS4, df.dH)[0, 1]); sl, ic = np.polyfit(df.Headcount_NFHS4, df.dH, 1)
R["conv"] = dict(r=r, slope=float(sl), intercept=float(ic), r2=r * r)
# states
ct = pd.crosstab(df.State, df.Tier).reindex(columns=names, fill_value=0); ct["n"] = ct.sum(axis=1)
sm = df.groupby("State").Headcount_NFHS5.mean(); ct["H5"] = sm
R["state_tab"] = {s: {k: float(v) for k, v in row.items()} for s, row in ct.iterrows()}
R["top10"] = df.nlargest(10, "Headcount_NFHS5")[["District", "State", "Headcount_NFHS4", "Headcount_NFHS5", "Intensity_NFHS5"]].round(2).to_dict("records")
R["low5_prog"] = df[df.Cluster == 0].nlargest(8, "dH")[["District", "State", "Headcount_NFHS4", "Headcount_NFHS5", "dH"]].round(2).to_dict("records")
R["worse"] = df[df.dH > 0][["District", "State", "dH"]].round(2).to_dict("records")
R["high_state_share"] = (ct.High / ct.High.sum()).sort_values(ascending=False).head(5).round(3).to_dict()
# sensitivity to feature sets
sets = {"Main: H4,A4,H5,A5": F4, "H4,H5 only": ["Headcount_NFHS4", "Headcount_NFHS5"], "H5,A5 only": ["Headcount_NFHS5", "Intensity_NFHS5"], "H5 only": ["Headcount_NFHS5"], "H5,A5,MPI5": ["Headcount_NFHS5", "Intensity_NFHS5", "MPI_NFHS5"], "H4,A4,H5,A5,dH": F4 + ["dH"]}
sens = {}
for nm, fs in sets.items():
    Xs = StandardScaler().fit_transform(df[fs]); l = KMeans(K, n_init=20, random_state=SEED).fit_predict(Xs)
    sens[nm] = dict(Sil=round(float(silhouette_score(Xs, l)), 3), ARI=round(float(adjusted_rand_score(lab_km, l)), 3))
R["sens"] = sens
# extension: NFHS-5 only on all districts
ext = raw.dropna(subset=["Headcount_NFHS5", "Intensity_NFHS5"]).reset_index(drop=True)
Xe = StandardScaler().fit_transform(ext[["Headcount_NFHS5", "Intensity_NFHS5"]]); ke = KMeans(3, n_init=50, random_state=SEED).fit(Xe)
oe = ext.assign(l=ke.labels_).groupby("l").Headcount_NFHS5.mean().sort_values(ascending=False).index.tolist(); me = {o: i for i, o in enumerate(oe)}; ext["Cluster"] = [me[l] for l in ke.labels_]
ov = ext.merge(df[["State", "District", "Cluster"]], on=["State", "District"], suffixes=("_ext", "_main"))
R["ext"] = dict(n=len(ext), sil=float(silhouette_score(Xe, ext.Cluster.map({0: 0, 1: 1, 2: 2}))), ari=float(adjusted_rand_score(ov.Cluster_ext, ov.Cluster_main)), agree=float((ov.Cluster_ext == ov.Cluster_main).mean()), sizes=ext.Cluster.value_counts().sort_index().tolist())
newd = ext[ext.State.isin(raw.State) & ~ext.set_index(["State", "District"]).index.isin(df.set_index(["State", "District"]).index)]
R["ext"]["new_n"] = len(newd); R["ext"]["new_sizes"] = newd.Cluster.value_counts().sort_index().reindex([0, 1, 2], fill_value=0).tolist()
df.to_csv("results/districts_with_clusters.csv", index=False); ext.to_csv("results/districts_nfhs5_only_clusters.csv", index=False)
json.dump(R, open("results/results.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ["n", "n_extracted", "n_excluded_nfhs4", "cluster_sizes", "methods", "prof", "sens", "stab", "conv", "ext", "sil_by_cluster", "sil_neg_share", "improved_share", "n_worse", "high_state_share"]}, indent=1, default=float))
print(sel.round(3).to_string()); print(R["pca_ev"], R["pca_load"]); print(R["top10"][:3]); print(R["worse"][:10])
