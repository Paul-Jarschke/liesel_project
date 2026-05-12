# ==============================================================================
# Hierarchical MNL — Margarine Dataset — 5-Component Mixture
# Matches export format of bayesm_2comp_mixture.R so that
# load_and_format_draws() in the marginal-density comparison script
# works identically.
#
# Model spec
#   Alternatives : 4  (BB, FL, SB, CH)
#   Params       : 5  (3 ASCs + price + feature)
#   Demographics : 5  (Income, Fs3_4, Fs5_, college, wgtS — all centred)
#   Components   : 5
# ==============================================================================

library(bayesm)
library(this.path)

# ── 1. Load data ───────────────────────────────────────────────────────────────
cat("Loading margarine data...\n")
script_dir <- this.path::here()
data(margarine)

choice_df <- margarine$choiceAtt # hhid, choice, price.*, feat.*
demo_df <- margarine$demos # hhid, Income, Fs3_4, Fs5_, college, wgtS

# ── 2. Scalars ─────────────────────────────────────────────────────────────────
n_alts <- 4L # BB FL SB CH
n_ascs <- n_alts - 1L # 3 ASCs (alt 1 = base)
n_cont <- 2L # price, feature
n_params <- n_ascs + n_cont # 5
n_comp <- 5L

alt_names <- c("BB", "FL", "SB", "CH")
param_names <- c(paste0("ASC_", alt_names[-1]), "price", "feature")

# ── 3. Demographics (Z) — centre each column ──────────────────────────────────
demo_vars <- c("Income", "Fs3_4", "Fs5_", "college", "wgtS")
n_demos <- length(demo_vars)
demo_names <- demo_vars

hh_ids <- sort(unique(choice_df$hhid))
n_units <- length(hh_ids)

Z_raw <- as.matrix(demo_df[match(hh_ids, demo_df$hhid), demo_vars])
Z <- scale(Z_raw, center = TRUE, scale = FALSE) # mean-centre only
cat("Z column means (should be ~0):", round(colMeans(Z), 8), "\n")

# ── 4. Build lgtdata ───────────────────────────────────────────────────────────
# For each household × occasion, X_it is (n_alts × n_params):
#   cols 1-3 : ASC indicators (0/1)
#   col  4   : price of chosen/unchosen alt
#   col  5   : feature flag of chosen/unchosen alt
price_cols <- paste0("price.", alt_names) # price.BB price.FL price.SB price.CH
feature_cols <- paste0("feat.", alt_names) # feat.BB  feat.FL  feat.SB  feat.CH

lgtdata <- vector("list", n_units)

for (i in seq_along(hh_ids)) {
    hh <- hh_ids[i]
    rows <- choice_df[choice_df$hhid == hh, ]
    n_obs <- nrow(rows)

    y_i <- as.integer(rows$choice) # already 1-indexed in bayesm margarine

    # Stack X across observations: (n_obs * n_alts) × n_params
    X_blocks <- vector("list", n_obs)
    for (t in seq_len(n_obs)) {
        asc_block <- cbind(0, diag(n_ascs)) # (4 × 3) ASC design
        price_vec <- as.numeric(rows[t, price_cols]) # length-4
        feature_vec <- as.numeric(rows[t, feature_cols])

        X_blocks[[t]] <- cbind(asc_block, price_vec, feature_vec)
    }

    lgtdata[[i]] <- list(
        y = y_i,
        X = do.call(rbind, X_blocks) # (n_obs * n_alts) × n_params
    )
}

cat(sprintf(
    "lgtdata built: %d households | median %d obs/hh | %d params | %d alts\n",
    n_units,
    as.integer(median(sapply(lgtdata, function(x) length(x$y)))),
    n_params, n_alts
))

# ── 5. Priors ──────────────────────────────────────────────────────────────────
Prior <- list(
    ncomp = n_comp,
    Ad    = 0.01 * diag(n_demos * n_params), # Delta ~ N(0, Ad^{-1})
    nu    = n_params + 3L, # Wishart df
    V     = (n_params + 3L) * diag(n_params), # Wishart scale
    Amu   = 0.01, # mu_k | Sigma_k ~ N(0, Sigma_k / Amu)
    a     = rep(1, n_comp) # Dirichlet — diffuse / symmetric
)

# ── 6. MCMC settings ───────────────────────────────────────────────────────────
R_total <- 41000L
burn_in <- 1000L
keep_every <- 4L

Mcmc <- list(R = R_total, keep = keep_every, nprint = 500L)

# ── 7. Run ─────────────────────────────────────────────────────────────────────
data_list <- list(p = n_alts, lgtdata = lgtdata, Z = Z)

cat("\nRunning rhierMnlRwMixture (5-component, margarine)...\n")
set.seed(42)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ── 8. Post-processing — discard burn-in ──────────────────────────────────────
cat("\nProcessing posterior samples...\n")
thinned_burn_in <- burn_in %/% keep_every
R_draws <- length(out$nmix$compdraw)
keep_idx <- seq(thinned_burn_in + 1L, R_draws)
n_samples <- length(keep_idx)
cat(sprintf(
    "Total thinned draws: %d | Retained after burn-in: %d\n",
    R_draws, n_samples
))

# ── 9. Extract mu and cov per component ───────────────────────────────────────
mu_list <- vector("list", n_comp)
cov_list <- vector("list", n_comp)

for (k in seq_len(n_comp)) {
    mu_list[[k]] <- t(
        sapply(out$nmix$compdraw, function(x) x[[k]]$mu)
    )[keep_idx, , drop = FALSE] # (n_samples × n_params)

    cov_list[[k]] <- t(
        sapply(out$nmix$compdraw, function(x) as.vector(chol2inv(x[[k]]$rooti)))
    )[keep_idx, , drop = FALSE] # (n_samples × n_params²)
}

# ── 10. Flatten horizontally — mirrors simulated-data export ──────────────────
mu_flat <- do.call(cbind, mu_list) # (n_samples, K * n_params)
cov_flat <- do.call(cbind, cov_list) # (n_samples, K * n_params²)

pvec_raw <- out$nmix$probdraw
if (nrow(pvec_raw) == n_comp) pvec_raw <- t(pvec_raw)
pvec_flat <- pvec_raw[keep_idx, , drop = FALSE] # (n_samples, K)

mu_draws_df <- as.data.frame(mu_flat)
cov_draws_df <- as.data.frame(cov_flat)
pvec_draws_df <- as.data.frame(pvec_flat)

# Column names (load_and_format_draws doesn't require these, but they help)
colnames(mu_draws_df) <- paste0(
    rep(paste0("Comp_", seq_len(n_comp)), each = n_params), "_",
    rep(param_names, times = n_comp)
)
colnames(pvec_draws_df) <- paste0("Comp_", seq_len(n_comp))

# ── 11. Delta draws ────────────────────────────────────────────────────────────
delta_draws_df <- as.data.frame(out$Deltadraw[keep_idx, , drop = FALSE])

# ── 12. Unit-level beta draws — (n_units, n_samples, n_params) ────────────────
beta_reordered <- aperm(out$betadraw[, , keep_idx, drop = FALSE], c(1, 3, 2))

# ── 13. Posterior summary ─────────────────────────────────────────────────────
cat("\n=== Posterior Mixture Weights ===\n")
print(round(colMeans(pvec_draws_df), 4))

cat("\n=== Component Means (mu_k) ===\n")
for (k in seq_len(n_comp)) {
    cat(sprintf("\n-- Component %d --\n", k))
    post_mean <- colMeans(mu_list[[k]])
    names(post_mean) <- param_names
    print(round(post_mean, 4))
}

# ── 14. Export ─────────────────────────────────────────────────────────────────
export_dir <- file.path(script_dir, "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_path <- file.path(export_dir, "bayesm_output_margarine_5comp.RData")

save(
    mu_draws_df, # (n_samples, K * n_params)
    cov_draws_df, # (n_samples, K * n_params²)
    pvec_draws_df, # (n_samples, K)
    delta_draws_df, # (n_samples, n_demos * n_params)
    beta_reordered, # (n_units, n_samples, n_params)
    n_samples,
    n_units,
    n_params,
    n_comp,
    n_demos,
    param_names,
    demo_names,
    alt_names,
    file = save_path
)

cat(sprintf(
    "\nSUCCESS: %d posterior samples saved to\n  %s\n",
    n_samples, save_path
))
cat("\nObjects saved:\n")
cat("  mu_draws_df   : shape (", n_samples, ",", n_comp * n_params, ")\n")
cat("  cov_draws_df  : shape (", n_samples, ",", n_comp * n_params^2, ")\n")
cat("  pvec_draws_df : shape (", n_samples, ",", n_comp, ")\n")
cat("  delta_draws_df: shape (", n_samples, ",", n_demos * n_params, ")\n")
cat("  beta_reordered: shape (", n_units, "x", n_samples, "x", n_params, ")\n")
