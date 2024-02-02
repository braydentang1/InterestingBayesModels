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
    vector<lower=0>[N_customers] lambda;
    vector[N_customers] z;
    real p_logit_mu;
    real<lower=0> p_logit_sigma;
    real<lower=0> gamma_alpha;
    real<lower=0> gamma_beta;
}
transformed parameters {
    vector[N_customers] p;
    vector[N_customers] p_logit;
    p_logit = p_logit_mu + p_logit_sigma * z;
    p = inv_logit(p_logit);
}
model {
    // priors
    lambda ~ gamma(gamma_alpha, gamma_beta);
    gamma_alpha ~ normal(0, 2.5);
    gamma_beta ~ normal(0, 2.5);
    p_logit_sigma ~ student_t(3, 0, 1);
    p_logit_mu ~ student_t(3, 0, 1);
    z ~ std_normal();

    // loglikelihood
    if (prior_only != 1) {
      for (i in 1:N_customers) {
        target += bg_nbd_log_density(p[i], lambda[i], frequency[i], T_age[i], recency[i]);
      }
    }
}
generated quantities {
}