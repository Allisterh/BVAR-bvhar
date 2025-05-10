# VAR----------------------------------
test_that("Test for varlse forecast", {
  skip_on_cran()
  
  num_col <- 3
  fit_var <- var_lm(etf_vix[, 1:3], 2)
  fit_vhar <- vhar_lm(etf_vix[, 1:3])
  
  num_forecast <- 2
  pred_var <- predict(fit_var, num_forecast)
  pred_vhar <- predict(fit_vhar, num_forecast)
  
  expect_s3_class(pred_var, "predbvhar")
  expect_s3_class(pred_vhar, "predbvhar")

  expect_equal(
    nrow(pred_var$forecast),
    num_forecast
  )
  expect_equal(
    ncol(pred_var$forecast),
    num_col
  )
  expect_equal(
    nrow(pred_vhar$forecast),
    num_forecast
  )
  expect_equal(
    ncol(pred_vhar$forecast),
    num_col
  )
  
})

help_var_bayes_pred <- function(bayes_spec, cov_spec, sparse) {
  etf_train <- etf_vix[1:50, 1:2]

  set.seed(1)
  fit_test <- var_bayes(
    etf_train,
    p = 1,
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  predict(fit_test, 3, sparse = sparse)
}

test_that("Forecast - VAR-HS-LDLT", {
  skip_on_cran()

  test_pred_dense <- help_var_bayes_pred(set_horseshoe(), set_ldlt(), FALSE)
  test_pred_sparse <- help_var_bayes_pred(set_horseshoe(), set_ldlt(), TRUE)

  expect_s3_class(test_pred_dense, "predbvhar")

  expect_s3_class(test_pred_sparse, "predbvhar")
})

help_vhar_bayes_pred <- function(bayes_spec, cov_spec, sparse) {
  etf_train <- etf_vix[1:50, 1:2]

  set.seed(1)
  fit_test <- vhar_bayes(
    etf_train,
    num_iter = 3,
    num_burn = 0,
    coef_spec = bayes_spec,
    contem_spec = bayes_spec,
    cov_spec = cov_spec,
    include_mean = TRUE
  )
  set.seed(1)
  predict(fit_test, 3, sparse = sparse)
}

test_that("Forecast - VHAR-Minn-LDLT", {
  skip_on_cran()

  test_pred_dense <- help_vhar_bayes_pred(set_bvhar(), set_ldlt(), FALSE)
  test_pred_sparse <- help_vhar_bayes_pred(set_bvhar(), set_ldlt(), TRUE)

  expect_s3_class(test_pred_dense, "predbvhar")

  expect_s3_class(test_pred_sparse, "predbvhar")
})
#> Test passed 🌈