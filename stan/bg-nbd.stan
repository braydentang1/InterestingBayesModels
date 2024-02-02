functions {
    real bg_nbd_log_density(real p, real lambda, data int x, data int age, data int Tx) {
        real first_term;
        real second_term;

        first_term = x * log(1-p) + x * log(lambda) - lambda * age;
        if (x > 0) {
            second_term = log(p) + (x-1) * log(1-p) + x * log(lambda) - lambda * Tx;
            return log_sum_exp(first_term, second_term);
        } else {
            return first_term;
        }
    }
}
data {
    int<lower=0> prior_only;
    int<lower=0> N_customers;
    array[N_customers] int<lower=0> recency;
    array[N_customers] int<lower=0> frequency;
    array[N_customers] int<lower=0> T_age;

}
parameters {
    array[N_customers] real<lower=0, upper=1> p;
    array[N_customers] real<lower=0> lambda;
    real<lower=0> beta_a;
    real<lower=0> beta_b;
    real<lower=0> gamma_alpha;
    real<lower=0> gamma_beta;
}
transformed parameters {
}
model {
    // priors
    lambda ~ gamma(gamma_alpha, gamma_beta);
    p ~ beta(beta_a, beta_b);
    gamma_alpha ~ normal(0, 2.5);
    gamma_beta ~ normal(0, 2.5);
    beta_a ~ normal(1, 5);
    beta_b ~ normal(1, 5);

    // loglikelihood
    if (prior_only != 1) {
      for (i in 1:N_customers) {
        target += bg_nbd_log_density(p[i], lambda[i], frequency[i], T_age[i], recency[i]);
      }
    }
}
generated quantities {
   
}