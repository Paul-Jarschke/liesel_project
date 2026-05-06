# ==============================================================================
# Rossi et al. (2006) Section 5.5.3 — Margarine Hierarchical MNL
# K=1 and K=5 Normal Mixture Prior
# ==============================================================================
library(this.path)
library(bayesm)

data(margarine)

select <- c(1, 2, 3, 4, 5, 7)
chPr_raw <- as.matrix(margarine$choicePrice)
demos_raw <- as.matrix(margarine$demos)

chPr <- cbind(
    chPr_raw[, 1],
    chPr_raw[, 2],
    log(chPr_raw[, 2 + select])
)
chPr <- chPr[chPr[, 2] %in% select, , drop = FALSE]
chPr[chPr[, 2] == 7, 2] <- 6

# ── Build raw lgtdata ──────────────────────────────────────────────────────────
hhid_list <- unique(chPr[, 1])
p <- length(select)
lgtdata <- list()
keep_hhids <- c()
ind <- 1

for (i in seq_along(hhid_list)) {
    hh_data <- chPr[chPr[, 1] == hhid_list[i], , drop = FALSE]
    if (nrow(hh_data) >= 5) {
        y <- hh_data[, 2]
        X <- createX(
            p = p, na = 1, Xa = hh_data[, 3:8],
            nd = NULL, Xd = NULL, INT = TRUE, base = 1
        )
        lgtdata[[ind]] <- list(y = as.integer(y), X = X)
        keep_hhids[ind] <- hhid_list[i]
        ind <- ind + 1
    }
}
cat(sprintf("Households: %d\n", length(lgtdata)))

# ── Standardize X variables (paper Section 5.5.3) ─────────────────────────────
# "For this reason, we advocate standardizing the X variables."
# Standardize using the grand mean and SD computed across all observations,
# so that all X variables share a common scale and location — this is the
# condition the paper requires for A_mu = 1/16 to be meaningful.
X_all <- do.call(rbind, lapply(lgtdata, function(hh) hh$X))
X_mean <- colMeans(X_all)
X_sd <- apply(X_all, 2, sd)

lgtdata <- lapply(lgtdata, function(hh) {
    Xs <- sweep(sweep(hh$X, 2, X_mean, "-"), 2, X_sd, "/")
    list(y = hh$y, X = Xs)
})

# ── Hierarchy matrix Z ─────────────────────────────────────────────────────────
Z_raw <- NULL
for (id in keep_hhids) {
    Z_raw <- rbind(Z_raw, demos_raw[demos_raw[, 1] == id, c(2, 5), drop = FALSE])
}

Z <- matrix(NA, nrow = nrow(Z_raw), ncol = 2)
Z[, 1] <- log(Z_raw[, 1]) - mean(log(Z_raw[, 1]))
Z[, 2] <- Z_raw[, 2] - mean(Z_raw[, 2])

# ── Prior (paper Section 5.5.3) ───────────────────────────────────────────────
# "We then set A_mu to 1/16 or so rather than 1/100."
# "We will set the prior on Sigma to be relatively diffuse
#  by setting nu to nvar+3 and V = vI."
nvar <- p # 6
nZ <- ncol(Z) # 2
A_mu <- 1 / 16
nu <- nvar + 3
V <- nu * diag(nvar)
A_Delta <- A_mu * diag(nZ + 1)

# ── Export helper ──────────────────────────────────────────────────────────────
export_dir <- file.path(getwd(), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_draws <- function(out, burn_in, nvar, ncomp, tag, dir) {
    R_total <- length(out$nmix$compdraw)
    keep <- (burn_in + 1):R_total
    n <- length(keep)

    # detect actual dimensions from stored draws
    first <- out$nmix$compdraw[[keep[1]]]
    nvar <- length(first[[1]]$mu)
    ncomp <- length(first)

    # pvec
    raw <- out$nmix$pvec
    if (is.matrix(raw) && nrow(raw) >= max(keep)) {
        pvec_mat <- raw[keep, , drop = FALSE]
    } else {
        pvec_mat <- matrix(1.0, nrow = n, ncol = ncomp)
    }
    pvec_mat <- matrix(as.numeric(pvec_mat), nrow = n, ncol = ncol(pvec_mat))
    colnames(pvec_mat) <- paste0("comp", seq_len(ncol(pvec_mat)))

    # mu and sigma diagonal
    mu_mat <- matrix(NA_real_, nrow = n, ncol = ncomp * nvar)
    sd_mat <- matrix(NA_real_, nrow = n, ncol = ncomp * nvar)
    cnames <- paste0(
        rep(paste0("k", seq_len(ncomp)), each = nvar),
        "_v", rep(seq_len(nvar), ncomp)
    )

    for (i in seq_along(keep)) {
        draw <- out$nmix$compdraw[[keep[i]]]
        for (k in seq_len(ncomp)) {
            cols <- ((k - 1) * nvar + 1):(k * nvar)
            mu_mat[i, cols] <- as.numeric(draw[[k]]$mu)
            sd_mat[i, cols] <- diag(chol2inv(draw[[k]]$rooti))
        }
    }
    colnames(mu_mat) <- cnames
    colnames(sd_mat) <- cnames

    write.csv(pvec_mat, file.path(dir, paste0("pvec_", tag, ".csv")), row.names = FALSE)
    write.csv(mu_mat, file.path(dir, paste0("mu_", tag, ".csv")), row.names = FALSE)
    write.csv(sd_mat, file.path(dir, paste0("sigma_diag_", tag, ".csv")), row.names = FALSE)

    saveRDS(out$betadraw[, , keep],
        file = file.path(dir, paste0("betadraw_", tag, ".rds"))
    )

    cat(sprintf("  Saved %s: %d draws, ncomp=%d, nvar=%d\n", tag, n, ncomp, nvar))
}

# ── Run K=1 and K=5 ───────────────────────────────────────────────────────────
data_list <- list(p = p, lgtdata = lgtdata, Z = Z)

for (K in c(1, 5)) {
    cat(sprintf("\n--- K=%d ---\n", K))
    Prior <- list(ncomp = K, A = A_Delta, nu = nu, V = V)
    Mcmc <- list(R = 41000, keep = 1, nprint = 500)

    set.seed(123)
    out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

    save_draws(out,
        burn_in = 1000, nvar = nvar, ncomp = K,
        tag = paste0("K", K), dir = export_dir
    )
    rm(out)
    gc()
}

cat("\nDone:", export_dir, "\n")
