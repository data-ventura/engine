# Reference tick values from Wilkinson's extended algorithm, as implemented by its authors in R's
# `labeling` package (Talbot, Lin & Hanrahan 2010), for dv-render's tests.
#
# Usage: Rscript extended-ticks.R <out.json>
# Needs R with the `labeling` package; for example with micromamba:
#   micromamba create -p ./r -c conda-forge r-base r-labeling && ./r/bin/Rscript extended-ticks.R out.json
#
# Cases: hand-picked ranges (zero-crossing, tiny, huge, degenerate), then random ranges across 12
# orders of magnitude, each with a target tick count m from 2 to 10, with only.loose FALSE and TRUE.
# Defaults otherwise: Q = c(1, 5, 2, 2.5, 4, 3), w = c(0.25, 0.2, 0.5, 0.05).

library(labeling)
args <- commandArgs(trailingOnly = TRUE)
set.seed(20261113)

cases <- list(
  c(-15.8, 35.6), c(0, 100), c(0, 1), c(0, 2.5e6), c(1, 1.0000001), c(-1, 1),
  c(0.001, 0.0123), c(1e9, 3.7e9), c(-273.15, 56.7), c(17, 23), c(0.5, 0.55), c(-3e-4, 7e-4)
)
for (i in 1:250) {
  mag <- 10^runif(1, -6, 6)
  a <- runif(1, -1, 1) * mag
  b <- a + runif(1, 0.01, 3) * mag
  cases[[length(cases) + 1]] <- c(a, b)
}

num <- function(x) sprintf("%.17g", x)
out <- c()
for (cs in cases) {
  for (m in 2:10) {
    for (loose in c(FALSE, TRUE)) {
      t <- extended(cs[1], cs[2], m, only.loose = loose)
      out <- c(out, sprintf(
        '{"dmin":%s,"dmax":%s,"m":%d,"only_loose":%s,"ticks":[%s]}',
        num(cs[1]), num(cs[2]), m, tolower(as.character(loose)), paste(num(t), collapse = ",")
      ))
    }
  }
}
writeLines(c(
  sprintf('{"labeling_version":"%s","cases":[', as.character(packageVersion("labeling"))),
  paste(out, collapse = ",\n"),
  "]}"
), args[1])
