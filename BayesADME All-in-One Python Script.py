# BayesADME All-in-One Python Script
# This script combines all core modules of the BayesADME software.
# All comments are in English.

# --- Standard Library Imports ---
import yaml
import logging
import os
import sys
import time
import argparse
from collections import defaultdict
import numpy as np
import scipy.stats
import pandas as pd

# --- Code from utils.py ---
# Specific logger for utility functions
util_logger = logging.getLogger("BayesADME_Utils")

def sample_dirichlet(alpha_params):
    """
    Samples from a Dirichlet distribution.
    Uses np.random.dirichlet, which takes alpha (concentration parameters) as input.

    Args:
        alpha_params (array-like, shape (k,)): 
            Parameters of the Dirichlet distribution. Sum of alpha_params should be > 0.
            All elements of alpha_params should be > 0 for np.random.dirichlet.

    Returns:
        np.ndarray: A sample of shape (k,) from Dirichlet(alpha_params).
                    Returns a uniform distribution if input is problematic.
    """
    if not isinstance(alpha_params, np.ndarray):
        alpha = np.array(alpha_params, dtype=float) 
    else:
        alpha = alpha_params.astype(float)

    if np.any(alpha < 0):
        util_logger.error(f"Dirichlet alpha_params contains negative values: {alpha}. This is invalid.")
        return np.ones_like(alpha) / len(alpha) if len(alpha) > 0 else np.array([])

    processed_alpha = np.maximum(alpha, 1e-10) 

    if np.sum(processed_alpha) == 0 : 
        util_logger.warning(f"Sum of processed alpha_params for Dirichlet is zero: {processed_alpha}. Returning uniform.")
        return np.ones_like(processed_alpha) / len(processed_alpha) if len(processed_alpha) > 0 else np.array([])
    
    try:
        return np.random.dirichlet(processed_alpha)
    except ValueError as e: 
        util_logger.error(f"Error sampling from Dirichlet with processed_alpha={processed_alpha} (original alpha={alpha}): {e}")
        return processed_alpha / np.sum(processed_alpha)


def sample_inverse_gamma(shape, scale):
    """
    Samples from an Inverse Gamma distribution IG(shape, scale).
    'shape' is alpha (concentration parameter), 'scale' is beta (rate/scale parameter).
    The mean of IG(shape, scale) is scale / (shape - 1) for shape > 1.

    Args:
        shape (float): The shape parameter (alpha) of the Inverse Gamma distribution. Must be > 0.
        scale (float): The scale parameter (beta) of the Inverse Gamma distribution. Must be > 0.

    Returns:
        float: A sample from IG(shape, scale). Returns 1.0 as a fallback if parameters are invalid.
    """
    if shape <= 0 or scale <= 0:
        util_logger.warning(f"Attempting to sample Inverse Gamma with non-positive shape/scale: shape={shape}, scale={scale}. Returning fallback value 1.0.")
        return 1.0 
    try:
        return scipy.stats.invgamma.rvs(a=shape, scale=scale)
    except Exception as e: 
        util_logger.error(f"Error sampling from Inverse Gamma (shape={shape}, scale={scale}): {e}")
        if shape > 1:
            return scale / (shape - 1)
        return 1.0


def calculate_epistatic_interaction_term(X_k_centered_col, X_l_centered_col):
    """
    Calculates the epistatic interaction term for two SNPs, w_ikl = x'_ik * x'_il.
    This assumes X_k_centered_col and X_l_centered_col are already centered.

    Args:
        X_k_centered_col (np.ndarray): N x 1 array for centered genotypes of SNP k.
        X_l_centered_col (np.ndarray): N x 1 array for centered genotypes of SNP l.

    Returns:
        np.ndarray: N x 1 array representing the interaction term w_kl.
    """
    return X_k_centered_col * X_l_centered_col


def calculate_log_likelihood_mixture_component(y_res_for_snp_j, x_j_col_centered, sigma_sq_e, prior_variance_of_component):
    """
    Calculates the log of the marginal likelihood for the OLS estimate b_j, 
    given that SNP j's effect comes from a specific mixture component.

    Args:
        y_res_for_snp_j (np.ndarray): Residual phenotype vector (N x 1) for SNP j.
        x_j_col_centered (np.ndarray): Centered genotype vector for SNP j (N x 1).
        sigma_sq_e (float): Current estimate of residual variance.
        prior_variance_of_component (float): Variance of the specific mixture component for SNP j's effect.

    Returns:
        float: Log likelihood value (constants like log(2*pi) are omitted).
    """
    xjt_xj = np.dot(x_j_col_centered.T, x_j_col_centered)

    if xjt_xj < 1e-9: 
        return 0.0 if prior_variance_of_component == 0.0 else -np.inf

    b_j_numerator = np.dot(x_j_col_centered.T, y_res_for_snp_j)
    b_j = b_j_numerator / xjt_xj 
    
    var_b_j_from_data = sigma_sq_e / xjt_xj

    if var_b_j_from_data <= 1e-9: 
        if abs(b_j) < 1e-9 and prior_variance_of_component == 0.0: 
            return 0.0 
        return -np.inf 

    marginal_variance_of_b_j = prior_variance_of_component + var_b_j_from_data

    if marginal_variance_of_b_j <= 1e-9: 
        if abs(b_j) < 1e-9 : return 0.0 
        return -np.inf 

    log_L_component = -0.5 * np.log(marginal_variance_of_b_j) - 0.5 * (b_j**2 / marginal_variance_of_b_j)
    
    return log_L_component


def get_centered_genotypes(genotype_matrix_original, center=True, scale=False):
    """
    Centers and optionally scales a genotype matrix (Individuals x SNPs).
    Missing values (NaNs) should be imputed BEFORE this step.

    Args:
        genotype_matrix_original (np.ndarray): The N x M raw genotype matrix.
        center (bool): If True, subtracts column means.
        scale (bool): If True, divides each column by its standard deviation.

    Returns:
        np.ndarray: The processed N x M genotype matrix.
    """
    if not isinstance(genotype_matrix_original, np.ndarray):
        genotypes = np.array(genotype_matrix_original, dtype=float)
    else:
        genotypes = genotype_matrix_original.astype(float, copy=True) 
    
    if np.isnan(genotypes).any():
        util_logger.warning("NaNs found in get_centered_genotypes. Imputation should precede. Attempting mean imputation.")
        col_means_for_nan_fallback = np.nanmean(genotypes, axis=0)
        col_means_for_nan_fallback[np.isnan(col_means_for_nan_fallback)] = 0 
        for j_col in range(genotypes.shape[1]):
            nan_mask_for_col_j = np.isnan(genotypes[:, j_col])
            if np.any(nan_mask_for_col_j):
                genotypes[nan_mask_for_col_j, j_col] = col_means_for_nan_fallback[j_col]

    if center:
        column_means = np.mean(genotypes, axis=0) 
        genotypes = genotypes - column_means 

    if scale:
        column_std_devs = np.std(genotypes, axis=0) 
        non_zero_std_mask = column_std_devs != 0
        genotypes[:, non_zero_std_mask] = genotypes[:, non_zero_std_mask] / column_std_devs[non_zero_std_mask]
        
    return genotypes

def derive_dominance_genotypes_raw(additive_genotype_matrix, maf_threshold=0.01):
    """
    Derives raw dominance genotypes (0 for homozygotes, 1 for heterozygotes)
    from an additive genotype matrix. Output is NOT centered.

    Args:
        additive_genotype_matrix (np.ndarray): N x M matrix of additive genotypes (e.g., 0,1,2).
        maf_threshold (float): MAF threshold below which dominance codings are set to zero.

    Returns:
        np.ndarray: N x M matrix of raw dominance genotypes (0 or 1).
    """
    if not isinstance(additive_genotype_matrix, np.ndarray):
        A_geno_matrix = np.array(additive_genotype_matrix, dtype=float)
    else:
        A_geno_matrix = additive_genotype_matrix.astype(float, copy=False) 

    N_individuals, M_snps = A_geno_matrix.shape
    dominance_genotypes_raw_matrix = np.zeros((N_individuals, M_snps), dtype=float)

    for j_snp_idx in range(M_snps):
        snp_additive_column = A_geno_matrix[:, j_snp_idx]
        
        if np.isnan(snp_additive_column).any():
            util_logger.debug(f"SNP column {j_snp_idx} for dominance derivation contains NaNs. MAF may be affected.")
        
        is_heterozygote = np.round(snp_additive_column) == 1
        dominance_genotypes_raw_matrix[is_heterozygote, j_snp_idx] = 1.0
        
        mean_allele_value = np.nanmean(snp_additive_column) 
        
        if np.isnan(mean_allele_value): 
             p_allele_A2 = 0.5 
             util_logger.warning(f"SNP column {j_snp_idx} is all NaNs. Defaulting p_allele to 0.5 for MAF check.")
        else:
             p_allele_A2 = mean_allele_value / 2.0

        p_allele_A1 = 1.0 - p_allele_A2
        maf = min(p_allele_A2, p_allele_A1)

        if maf < maf_threshold:
            dominance_genotypes_raw_matrix[:, j_snp_idx] = 0.0 
            util_logger.debug(f"SNP {j_snp_idx} MAF {maf:.4f} < {maf_threshold}. Setting raw dominance to zeros.")
            
    return dominance_genotypes_raw_matrix

# --- Code from model_config.py ---
config_logger = logging.getLogger("BayesADME_Config")

class ModelConfig:
    """
    Manages the configuration parameters for the BayesADME model.
    Reads parameters from a YAML file and provides access to them.
    Includes sensible defaults for many parameters.
    """
    def __init__(self, config_file_path=None, config_dict=None):
        self.config = {}
        self._set_defaults() 

        if config_dict:
            config_logger.info("Loading configuration from provided dictionary.")
            self._update_config_from_dict(config_dict) 
        elif config_file_path:
            config_logger.info(f"Loading configuration from YAML file: {config_file_path}")
            if not os.path.exists(config_file_path):
                config_logger.error(f"Configuration file not found: {config_file_path}. Raising FileNotFoundError.")
                raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
            try:
                with open(config_file_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self._update_config_from_dict(yaml_config) 
                else:
                    config_logger.warning(f"Configuration file {config_file_path} is empty or invalid. Using defaults where applicable.")
            except yaml.YAMLError as e:
                config_logger.error(f"Error parsing YAML file {config_file_path}: {e}. Using defaults where applicable.")
        else:
            config_logger.info("No configuration file or dictionary provided. Using default parameters. Critical paths must be set later.")

        self._validate_config() 
        
    def _set_defaults(self):
        self.config = {
            "phenotype_file": None, "additive_genotype_file": None, "dominance_genotype_file": None, 
            "annotation_file": None, "derive_dominance_from_additive": False,
            "output_dir": "bayesadme_output", "output_prefix": "bayesadme_run",
            "num_iterations": 20000, "burn_in": 5000, "thinning": 10, "random_seed": None, 
            "include_additive": True, "include_dominance": True, "include_epistasis": True, 
            "additive_num_mixture_components": 4, 
            "prior_additive_variances_fixed": [0.0, 0.0001, 0.001, 0.01],
            "prior_additive_pi_alpha": 1.0, 
            "dominance_num_mixture_components": 3, 
            "prior_dominance_variances_fixed": [0.0, 0.0001, 0.001],
            "prior_dominance_pi_alpha": 1.0, 
            "prior_epistasis_pi0": 0.999, "prior_epistasis_variance_fixed": 0.001, 
            "epistasis_max_active_pairs": 1000, "epistasis_proposal_strategy": "random", 
            "epistasis_snp_pair_proposal_count": 100,
            "use_functional_annotations": False, "functional_prior_config": None,
            "num_annotation_categories": 1, 
            "center_genotypes": True, "scale_genotypes": False, 
            "missing_phenotype_value": "NA", "genotype_missing_fill_value": "mean", 
            "maf_threshold_for_dominance": 0.01, "parallel_cores": 1,
            "prior_sigma_e_nu0": 0.001, "prior_sigma_e_s0": 0.001
        }

    def _update_config_from_dict(self, source_dict):
        for key, value in source_dict.items():
            if key in self.config:
                self.config[key] = value
            else:
                config_logger.warning(f"Unknown configuration key '{key}' in source dictionary. It will be ignored.")

    def _validate_config(self):
        if not self.config.get("phenotype_file"):
            raise ValueError("Phenotype file path ('phenotype_file') must be specified.")
        if not self.config.get("additive_genotype_file"):
            raise ValueError("Additive genotype file path ('additive_genotype_file') must be specified.")

        if self.config.get("derive_dominance_from_additive") and self.config.get("dominance_genotype_file"):
            config_logger.warning("'derive_dominance_from_additive' is true, 'dominance_genotype_file' will be ignored.")
            self.config["dominance_genotype_file"] = None 

        if self.config.get("include_dominance") and \
           not self.config.get("derive_dominance_from_additive") and \
           not self.config.get("dominance_genotype_file"):
            config_logger.warning("Dominance included, but no file and not deriving. Provide file or set derivation to true for dominance effects.")

        if self.config.get("use_functional_annotations") and not self.config.get("annotation_file"):
            raise ValueError("Functional annotations enabled, but 'annotation_file' not provided.")

        if self.config.get("burn_in", 0) >= self.config.get("num_iterations", 1):
            raise ValueError("Burn-in period must be less than total MCMC iterations.")

        if len(self.config.get("prior_additive_variances_fixed", [])) != self.config.get("additive_num_mixture_components", 0):
            raise ValueError("'prior_additive_variances_fixed' length must match 'additive_num_mixture_components'.")
        if self.config.get("prior_additive_variances_fixed") and self.config.get("prior_additive_variances_fixed")[0] != 0.0:
            config_logger.warning("First 'prior_additive_variances_fixed' element should be 0.0 for zero-effect component.")

        if self.config.get("include_dominance"):
            if len(self.config.get("prior_dominance_variances_fixed", [])) != self.config.get("dominance_num_mixture_components", 0):
                raise ValueError("'prior_dominance_variances_fixed' length must match 'dominance_num_mixture_components'.")
            if self.config.get("prior_dominance_variances_fixed") and self.config.get("prior_dominance_variances_fixed")[0] != 0.0:
                config_logger.warning("First 'prior_dominance_variances_fixed' element should be 0.0.")
        
        config_logger.debug("Configuration (re-)validated successfully.")

    def log_config(self):
        config_logger.info("Current Model Configuration:")
        for key, value in sorted(self.config.items()):
            config_logger.info(f"  {key}: {value}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key not in self.config:
            raise KeyError(f"Configuration key '{key}' not found.")
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

# --- Code from data_handler.py ---
data_handler_logger = logging.getLogger("BayesADME_DataHandler")

class DataHandler:
    """
    Handles loading, preprocessing, validation, and alignment of input data 
    for the BayesADME model.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.phenotypes_raw = None; self.additive_genotypes_raw = None; self.dominance_genotypes_raw = None
        self.snp_annotations_raw_series = None
        self.individual_ids_pheno_raw = None; self.individual_ids_geno_raw = None
        self.snp_ids_geno_raw = None; self.snp_ids_annot_raw = None
        self.phenotypes_final = None; self.additive_genotypes_final = None; self.dominance_genotypes_final = None
        self.snp_annotations_mapped_final = None
        self.aligned_individual_ids_final = None; self.aligned_snp_ids_final = None
        self.functional_category_name_to_int_map = None 
        self.num_functional_categories = 1
        self.N_final = 0; self.M_final = 0 

    def load_and_process_data(self):
        data_handler_logger.info("Starting data loading and processing pipeline...")
        self._load_phenotypes_raw()
        self._load_additive_genotypes_raw()

        if self.config.get("include_dominance"):
            if self.config.get("derive_dominance_from_additive"):
                if self.additive_genotypes_raw is not None:
                    data_handler_logger.info("Deriving raw dominance genotypes from raw additive genotypes.")
                    self.dominance_genotypes_raw = derive_dominance_genotypes_raw(
                        self.additive_genotypes_raw, 
                        self.config.get("maf_threshold_for_dominance", 0.01)
                    )
                else: 
                    data_handler_logger.error("Critical: Cannot derive dominance as raw additive genotypes failed to load.")
            elif self.config.get("dominance_genotype_file"):
                self._load_dominance_genotypes_raw()
        
        if self.config.get("use_functional_annotations") and self.config.get("annotation_file"):
            self._load_annotations_raw()

        self._impute_missing_genotypes_in_raw_matrices()
        self._align_data_across_sources() 
        self._process_functional_annotations_for_aligned_snps() 
        self._preprocess_final_genotype_matrices() 
        
        if self.N_final == 0 or self.M_final == 0:
            raise ValueError(f"Data processing resulted in N={self.N_final} individuals or M={self.M_final} SNPs. Check inputs.")

        data_handler_logger.info(f"Data loading complete. Final N={self.N_final} individuals, M={self.M_final} SNPs.")
        data_handler_logger.info(f"Functional categories: {self.num_functional_categories}")

        return {
            "phenotypes": self.phenotypes_final, "additive_genotypes": self.additive_genotypes_final, 
            "dominance_genotypes": self.dominance_genotypes_final, 
            "snp_annotations_mapped": self.snp_annotations_mapped_final,
            "num_functional_categories": self.num_functional_categories,
            "N": self.N_final, "M": self.M_final,
            "aligned_snp_ids": self.aligned_snp_ids_final,
            "aligned_individual_ids": self.aligned_individual_ids_final
        }

    def _read_csv_or_tsv_with_error_handling(self, file_path, **kwargs):
        if not os.path.exists(file_path):
            data_handler_logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Required input file not found: {file_path}")
        try:
            return pd.read_csv(file_path, sep=None, engine='python', **kwargs)
        except Exception as e:
            data_handler_logger.error(f"Error reading file {file_path}: {e}")
            raise 

    def _load_phenotypes_raw(self):
        pheno_file = self.config.get("phenotype_file")
        data_handler_logger.info(f"Loading raw phenotypes from: {pheno_file}")
        df_pheno = self._read_csv_or_tsv_with_error_handling(pheno_file, header=0)
        
        if df_pheno.shape[1] < 2: 
            raise ValueError("Phenotype file must contain at least two columns (IndividualID, PhenotypeValue).")
        
        self.individual_ids_pheno_raw = df_pheno.iloc[:, 0].astype(str).tolist()
        self.phenotypes_raw = df_pheno.iloc[:, 1].values 

        missing_pheno_config_val = self.config.get("missing_phenotype_value", "NA")
        if not (missing_pheno_config_val.upper() == "NA" or missing_pheno_config_val.upper() == "NAN"):
            try:
                missing_float_val = float(missing_pheno_config_val)
                self.phenotypes_raw[self.phenotypes_raw == missing_float_val] = np.nan
            except ValueError: pass 
        
        num_missing_phenos = pd.Series(self.phenotypes_raw).isnull().sum()
        data_handler_logger.info(f"Loaded {len(self.phenotypes_raw)} raw phenotype records. Found {num_missing_phenos} missing values (as np.nan).")

    def _load_genotypes_raw_common(self, file_path, is_additive_file_type):
        data_handler_logger.info(f"Loading raw {'additive' if is_additive_file_type else 'dominance'} genotypes from: {file_path}")
        df_geno = self._read_csv_or_tsv_with_error_handling(file_path, header=0, index_col=0)
        
        individual_ids_from_file = df_geno.index.astype(str).tolist()
        snp_ids_from_file = df_geno.columns.astype(str).tolist()
        genotype_matrix_from_file = df_geno.values.astype(float)

        if is_additive_file_type:
            self.additive_genotypes_raw = genotype_matrix_from_file
            self.individual_ids_geno_raw = individual_ids_from_file
            self.snp_ids_geno_raw = snp_ids_from_file
        else: 
            self.dominance_genotypes_raw = genotype_matrix_from_file
            if self.snp_ids_geno_raw and self.snp_ids_geno_raw != snp_ids_from_file:
                data_handler_logger.warning("SNP IDs in loaded dominance file differ from additive. Alignment critical.")
            if self.individual_ids_geno_raw and self.individual_ids_geno_raw != individual_ids_from_file:
                data_handler_logger.warning("Individual IDs in loaded dominance file differ from additive. Alignment critical.")
        
        data_handler_logger.info(f"Loaded raw {'additive' if is_additive_file_type else 'dominance'} genotypes for "
                    f"{genotype_matrix_from_file.shape[0]} inds, {genotype_matrix_from_file.shape[1]} SNPs from {file_path}.")

    def _load_additive_genotypes_raw(self):
        self._load_genotypes_raw_common(self.config.get("additive_genotype_file"), is_additive_file_type=True)

    def _load_dominance_genotypes_raw(self):
        self._load_genotypes_raw_common(self.config.get("dominance_genotype_file"), is_additive_file_type=False)

    def _load_annotations_raw(self):
        annot_file_path = self.config.get("annotation_file")
        data_handler_logger.info(f"Loading raw SNP functional annotations from: {annot_file_path}")
        df_annot = self._read_csv_or_tsv_with_error_handling(annot_file_path, header=0)
        
        if df_annot.shape[1] < 2: 
            raise ValueError("Annotation file must contain at least two columns (SNP_ID, CategoryName).")
        
        self.snp_ids_annot_raw = df_annot.iloc[:, 0].astype(str).tolist()
        self.snp_annotations_raw_series = pd.Series(
            df_annot.iloc[:, 1].astype(str).values, 
            index=self.snp_ids_annot_raw
        )
        data_handler_logger.info(f"Loaded raw annotations for {len(self.snp_annotations_raw_series)} SNPs.")

    def _impute_missing_genotypes_in_raw_matrices(self):
        matrices_to_process = []
        if self.additive_genotypes_raw is not None: matrices_to_process.append("additive")
        if self.dominance_genotypes_raw is not None: matrices_to_process.append("dominance")

        fill_strategy = self.config.get("genotype_missing_fill_value", "mean")
        
        for matrix_type_str in matrices_to_process:
            current_matrix = getattr(self, f"{matrix_type_str}_genotypes_raw")
            
            if np.isnan(current_matrix).any(): 
                data_handler_logger.info(f"Imputing NaNs in raw {matrix_type_str} genotypes using strategy: '{fill_strategy}'...")
                if fill_strategy == "mean":
                    column_means_for_imputation = np.nanmean(current_matrix, axis=0)
                    column_means_for_imputation[np.isnan(column_means_for_imputation)] = 0.0
                    nan_indices = np.where(np.isnan(current_matrix))
                    current_matrix[nan_indices] = np.take(column_means_for_imputation, nan_indices[1])
                elif fill_strategy == "zero":
                    current_matrix[np.isnan(current_matrix)] = 0.0
                else: 
                    try:
                        fill_numeric_value = float(fill_strategy)
                        current_matrix[np.isnan(current_matrix)] = fill_numeric_value
                    except ValueError:
                        data_handler_logger.warning(f"Invalid 'genotype_missing_fill_value': {fill_strategy}. Defaulting to zero-imputation for {matrix_type_str}.")
                        current_matrix[np.isnan(current_matrix)] = 0.0
            setattr(self, f"{matrix_type_str}_genotypes_raw", current_matrix)

    def _align_data_across_sources(self):
        if self.phenotypes_raw is None or self.additive_genotypes_raw is None:
            raise ValueError("Critical: Cannot align. Raw phenotype or additive genotype data not loaded.")

        pheno_series_raw = pd.Series(self.phenotypes_raw, index=self.individual_ids_pheno_raw)
        add_geno_df_raw = pd.DataFrame(self.additive_genotypes_raw, 
                                       index=self.individual_ids_geno_raw, 
                                       columns=self.snp_ids_geno_raw)

        pheno_series_valid = pheno_series_raw.dropna() 
        if pheno_series_valid.empty: 
            raise ValueError("No individuals left after removing missing phenotypes.")
        
        common_individual_ids = sorted(list(set(pheno_series_valid.index).intersection(set(add_geno_df_raw.index))))
        if not common_individual_ids: 
            raise ValueError("No common individuals between phenotypes and genotypes after filtering.")
        
        self.N_final = len(common_individual_ids)
        self.aligned_individual_ids_final = common_individual_ids
        data_handler_logger.info(f"Found {self.N_final} common individuals with valid phenotypes.")
        
        self.phenotypes_final = pheno_series_valid.loc[self.aligned_individual_ids_final].values.reshape(-1, 1)
        self.additive_genotypes_final = add_geno_df_raw.loc[self.aligned_individual_ids_final, :].copy() 

        if self.dominance_genotypes_raw is not None:
            dom_geno_df_raw = pd.DataFrame(self.dominance_genotypes_raw,
                                           index=self.individual_ids_geno_raw, 
                                           columns=self.snp_ids_geno_raw)      
            self.dominance_genotypes_final = dom_geno_df_raw.loc[self.aligned_individual_ids_final, :].copy()
        else:
            self.dominance_genotypes_final = None 

        current_snp_ids_in_additive = list(self.additive_genotypes_final.columns)
        
        if self.config.get("use_functional_annotations") and self.snp_annotations_raw_series is not None:
            common_snp_ids_for_model = sorted(list(set(current_snp_ids_in_additive).intersection(set(self.snp_annotations_raw_series.index))))
            if not common_snp_ids_for_model:
                data_handler_logger.warning("No common SNPs between genotypes and annotations. Annotations effectively ignored.")
                self.aligned_snp_ids_final = current_snp_ids_in_additive
            else:
                data_handler_logger.info(f"Found {len(common_snp_ids_for_model)} common SNPs between genotypes and annotations.")
                self.aligned_snp_ids_final = common_snp_ids_for_model
        else: 
            self.aligned_snp_ids_final = current_snp_ids_in_additive

        self.M_final = len(self.aligned_snp_ids_final)
        if self.M_final == 0:
            raise ValueError("No SNPs remain after alignment.")

        self.additive_genotypes_final = self.additive_genotypes_final.loc[:, self.aligned_snp_ids_final]
        if self.dominance_genotypes_final is not None:
            self.dominance_genotypes_final = self.dominance_genotypes_final.loc[:, self.aligned_snp_ids_final]
        
        self.additive_genotypes_final = self.additive_genotypes_final.values
        if self.dominance_genotypes_final is not None:
            self.dominance_genotypes_final = self.dominance_genotypes_final.values

        data_handler_logger.info(f"Data alignment complete. Final N={self.N_final} individuals, M={self.M_final} SNPs.")

    def _process_functional_annotations_for_aligned_snps(self):
        if not self.config.get("use_functional_annotations") or self.snp_annotations_raw_series is None or self.aligned_snp_ids_final is None:
            self.snp_annotations_mapped_final = np.zeros(self.M_final, dtype=int)
            self.num_functional_categories = 1
            self.functional_category_name_to_int_map = {"default_category_0": 0}
            self.config.config['num_annotation_categories'] = 1 
            data_handler_logger.info("Functional annotations not used/available. Using single default category.")
            return

        aligned_annotations_series_raw = self.snp_annotations_raw_series.reindex(self.aligned_snp_ids_final)
        default_category_for_missing_snps = "unknown_or_default_annotation"
        aligned_annotations_series_raw.fillna(default_category_for_missing_snps, inplace=True)

        unique_category_names = aligned_annotations_series_raw.unique()
        self.functional_category_name_to_int_map = {name: i for i, name in enumerate(unique_category_names)}
        self.num_functional_categories = len(unique_category_names)
        self.config.config['num_annotation_categories'] = self.num_functional_categories

        self.snp_annotations_mapped_final = aligned_annotations_series_raw.map(self.functional_category_name_to_int_map).values
        data_handler_logger.info(f"Processed functional annotations for {self.M_final} SNPs. Found {self.num_functional_categories} categories.")
        data_handler_logger.debug(f"Functional category map: {self.functional_category_name_to_int_map}")

    def _preprocess_final_genotype_matrices(self):
        if self.additive_genotypes_final is not None:
            data_handler_logger.info("Applying centering/scaling to final additive genotype matrix...")
            self.additive_genotypes_final = get_centered_genotypes(
                self.additive_genotypes_final,
                center=self.config.get("center_genotypes", True),
                scale=self.config.get("scale_genotypes", False)
            )
        
        if self.dominance_genotypes_final is not None and self.config.get("include_dominance"):
            data_handler_logger.info("Applying centering/scaling to final dominance genotype matrix...")
            self.dominance_genotypes_final = get_centered_genotypes(
                self.dominance_genotypes_final, 
                center=self.config.get("center_genotypes", True),
                scale=False 
            )
        elif self.config.get("include_dominance") and self.dominance_genotypes_final is None:
            data_handler_logger.warning("Dominance included, but matrix unavailable. Dominance effects cannot be estimated. Using zero matrix if model requires.")
            self.dominance_genotypes_final = np.zeros((self.N_final, self.M_final))

# --- Code from samplers.py ---
samplers_logger = logging.getLogger("BayesADME_Samplers")

def sample_mu_overall(y_residuals_excluding_mu, current_sigma_sq_e, num_individuals):
    if num_individuals == 0: 
        samplers_logger.warning("Cannot sample mu: num_individuals is 0.")
        return 0.0 
    
    posterior_mean_mu = np.mean(y_residuals_excluding_mu)
    posterior_variance_mu = current_sigma_sq_e / num_individuals
    
    if posterior_variance_mu <= 1e-9: 
        samplers_logger.debug(f"Variance for mu sampling near zero ({posterior_variance_mu}). Returning mean_mu {posterior_mean_mu}.")
        return posterior_mean_mu 
        
    return np.random.normal(posterior_mean_mu, np.sqrt(posterior_variance_mu))

def sample_residual_variance(current_total_residuals, num_individuals, prior_nu0=0.001, prior_s0=0.001):
    if num_individuals == 0: 
        samplers_logger.warning("Cannot sample residual variance: num_individuals is 0.")
        return 1.0 

    sse = np.dot(current_total_residuals.T, current_total_residuals)
    sse_scalar = sse.item() if isinstance(sse, np.ndarray) and sse.ndim > 0 else float(sse)

    posterior_shape = prior_nu0 + num_individuals / 2.0
    posterior_scale = prior_s0 + sse_scalar / 2.0
    
    sampled_variance = sample_inverse_gamma(posterior_shape, posterior_scale)
    return max(sampled_variance, 1e-9) 

def sample_component_mixture_proportions(
    counts_in_components_per_category, 
    prior_alpha_config, 
    num_functional_categories_model
    ):
    num_cats_from_counts_matrix, num_mixture_components = counts_in_components_per_category.shape
    if num_cats_from_counts_matrix != num_functional_categories_model:
         raise ValueError(f"Mismatch in num_categories: from counts ({num_cats_from_counts_matrix}), from param ({num_functional_categories_model})")

    sampled_pis_all_categories = np.zeros_like(counts_in_components_per_category, dtype=float)

    for cat_idx in range(num_functional_categories_model):
        counts_for_this_category = counts_in_components_per_category[cat_idx, :]
        dirichlet_params_for_this_category = np.zeros(num_mixture_components, dtype=float)

        if isinstance(prior_alpha_config, (float, int)):
            dirichlet_params_for_this_category = np.full(num_mixture_components, float(prior_alpha_config)) + counts_for_this_category
        elif isinstance(prior_alpha_config, np.ndarray):
            if prior_alpha_config.ndim == 1 and len(prior_alpha_config) == num_mixture_components:
                dirichlet_params_for_this_category = prior_alpha_config + counts_for_this_category
            elif prior_alpha_config.ndim == 2 and prior_alpha_config.shape == (num_functional_categories_model, num_mixture_components):
                dirichlet_params_for_this_category = prior_alpha_config[cat_idx, :] + counts_for_this_category
            else:
                raise ValueError("prior_alpha_config np.ndarray shape not understood.")
        else:
            raise TypeError("prior_alpha_config must be a float, int, or a NumPy array.")
        
        sampled_pis_all_categories[cat_idx, :] = sample_dirichlet(dirichlet_params_for_this_category)
        
    return sampled_pis_all_categories

def sample_single_snp_effect_from_mixture(
    y_residuals_for_snp_j, x_j_genotype_column, current_sigma_sq_e,
    mixture_proportions_for_category, prior_variances_for_category, 
    snp_global_index 
    ):
    num_mixture_components = len(mixture_proportions_for_category)
    log_posterior_probs_unnormalized = np.zeros(num_mixture_components)

    xjt_xj = np.dot(x_j_genotype_column.T, x_j_genotype_column)

    if xjt_xj < 1e-9: 
        if prior_variances_for_category[0] != 0.0:
            samplers_logger.warning(f"SNP {snp_global_index} xjt_xj~0, but category's first prior var is not 0.0.")
        return 0.0, 0 
    
    b_j_numerator = np.dot(x_j_genotype_column.T, y_residuals_for_snp_j)
    b_j = b_j_numerator / xjt_xj 
    b_j = b_j.item() if isinstance(b_j, np.ndarray) and b_j.ndim == 0 else float(b_j)

    for k_component_idx in range(num_mixture_components):
        log_prior_prob_k = np.log(mixture_proportions_for_category[k_component_idx] + 1e-12) 
        log_marginal_likelihood_k = calculate_log_likelihood_mixture_component(
            y_residuals_for_snp_j, x_j_genotype_column, current_sigma_sq_e, 
            prior_variances_for_category[k_component_idx]
        )
        log_posterior_probs_unnormalized[k_component_idx] = log_prior_prob_k + log_marginal_likelihood_k
    
    max_log_prob = np.max(log_posterior_probs_unnormalized)
    log_posterior_probs_unnormalized -= max_log_prob 
    posterior_probs_normalized = np.exp(log_posterior_probs_unnormalized)
    sum_of_probs = np.sum(posterior_probs_normalized)
    
    if sum_of_probs == 0 or np.isnan(sum_of_probs) or np.isinf(sum_of_probs):
        samplers_logger.warning(f"Posterior probs for SNP {snp_global_index} invalid (sum={sum_of_probs}). Assigning to zero component.")
        zero_variance_component_indices = np.where(prior_variances_for_category == 0.0)[0]
        chosen_component_idx = zero_variance_component_indices[0] if len(zero_variance_component_indices) > 0 else 0
        return 0.0, chosen_component_idx 
        
    posterior_probs_normalized /= sum_of_probs
    
    try:
        chosen_component_idx = np.random.choice(num_mixture_components, p=posterior_probs_normalized)
    except ValueError as e: 
        samplers_logger.warning(f"Error sampling component for SNP {snp_global_index} (probs sum: {np.sum(posterior_probs_normalized)}): {e}. Assigning to zero component.")
        zero_variance_component_indices = np.where(prior_variances_for_category == 0.0)[0]
        chosen_component_idx = zero_variance_component_indices[0] if len(zero_variance_component_indices) > 0 else 0

    new_sampled_effect_j = 0.0
    prior_variance_of_chosen_component = prior_variances_for_category[chosen_component_idx]

    if prior_variance_of_chosen_component > 1e-9: 
        inverse_of_chosen_prior_variance = 1.0 / prior_variance_of_chosen_component
        posterior_variance_effect_j = 1.0 / ( (xjt_xj / current_sigma_sq_e) + inverse_of_chosen_prior_variance )
        posterior_mean_effect_j = posterior_variance_effect_j * (b_j_numerator / current_sigma_sq_e)
        posterior_mean_effect_j = posterior_mean_effect_j.item() if isinstance(posterior_mean_effect_j, np.ndarray) and posterior_mean_effect_j.ndim == 0 else float(posterior_mean_effect_j)

        if posterior_variance_effect_j < 0: 
            samplers_logger.warning(f"Negative posterior variance ({posterior_variance_effect_j:.3e}) for SNP {snp_global_index}. Setting effect to 0.")
            new_sampled_effect_j = 0.0
            zero_variance_component_indices = np.where(prior_variances_for_category == 0.0)[0] 
            chosen_component_idx = zero_variance_component_indices[0] if len(zero_variance_component_indices) > 0 else 0
        else:
            new_sampled_effect_j = np.random.normal(posterior_mean_effect_j, np.sqrt(posterior_variance_effect_j))
    else: 
        new_sampled_effect_j = 0.0
        if prior_variance_of_chosen_component != 0.0: 
            zero_variance_component_indices = np.where(prior_variances_for_category == 0.0)[0]
            if len(zero_variance_component_indices) > 0:
                chosen_component_idx = zero_variance_component_indices[0]
    return new_sampled_effect_j, chosen_component_idx

def sample_single_epistatic_effect(
    y_residuals_for_pair_kl, w_kl_interaction_column, current_sigma_sq_e,
    prior_prob_epistasis_is_zero, prior_variance_epistasis_if_nonzero,
    snp_pair_indices_tuple 
    ):
    wT_w = np.dot(w_kl_interaction_column.T, w_kl_interaction_column)

    if wT_w < 1e-9: 
        return 0.0, 0 

    b_I_numerator = np.dot(w_kl_interaction_column.T, y_residuals_for_pair_kl)
    b_I = b_I_numerator / wT_w
    b_I = b_I.item() if isinstance(b_I, np.ndarray) and b_I.ndim == 0 else float(b_I)

    variance_b_I_if_gamma0 = current_sigma_sq_e / wT_w
    if variance_b_I_if_gamma0 <= 1e-9: 
        log_L0 = -np.inf if abs(b_I) > 1e-9 else 0.0 
    else: 
        log_L0 = -0.5 * np.log(variance_b_I_if_gamma0) - 0.5 * (b_I**2 / variance_b_I_if_gamma0)

    variance_b_I_if_gamma1 = prior_variance_epistasis_if_nonzero + variance_b_I_if_gamma0
    if variance_b_I_if_gamma1 <= 1e-9:
        log_L1 = -np.inf if abs(b_I) > 1e-9 else 0.0
    else:
        log_L1 = -0.5 * np.log(variance_b_I_if_gamma1) - 0.5 * (b_I**2 / variance_b_I_if_gamma1)
    
    log_prior_odds_gamma1_vs_gamma0 = np.log( 
        (1.0 - prior_prob_epistasis_is_zero + 1e-12) / (prior_prob_epistasis_is_zero + 1e-12) 
    ) 
    log_likelihood_ratio_L1_vs_L0 = log_L1 - log_L0
    
    if np.isinf(log_likelihood_ratio_L1_vs_L0) and log_likelihood_ratio_L1_vs_L0 < 0: 
        log_posterior_odds_gamma1_vs_gamma0 = -np.inf
    elif np.isinf(log_likelihood_ratio_L1_vs_L0) and log_likelihood_ratio_L1_vs_L0 > 0: 
        log_posterior_odds_gamma1_vs_gamma0 = np.inf
    elif np.isnan(log_likelihood_ratio_L1_vs_L0): 
        samplers_logger.warning(f"Log likelihood ratio for epistatic pair {snp_pair_indices_tuple} is NaN. Defaulting posterior odds.")
        log_posterior_odds_gamma1_vs_gamma0 = -np.inf 
    else:
        log_posterior_odds_gamma1_vs_gamma0 = log_prior_odds_gamma1_vs_gamma0 + log_likelihood_ratio_L1_vs_L0
    
    prob_gamma_kl_is_1 = 1.0 / (1.0 + np.exp(-log_posterior_odds_gamma1_vs_gamma0)) 
    
    if np.isnan(prob_gamma_kl_is_1): 
        samplers_logger.warning(f"NaN in epistatic inclusion probability for pair {snp_pair_indices_tuple}. Defaulting to 0.")
        prob_gamma_kl_is_1 = 0.0

    inclusion_indicator_gamma_kl = 1 if np.random.rand() < prob_gamma_kl_is_1 else 0

    new_sampled_I_kl = 0.0
    if inclusion_indicator_gamma_kl == 1: 
        inverse_prior_variance_epistasis_nonzero = 1.0 / prior_variance_epistasis_if_nonzero
        posterior_variance_I_kl = 1.0 / ( (wT_w / current_sigma_sq_e) + inverse_prior_variance_epistasis_nonzero )
        posterior_mean_I_kl = posterior_variance_I_kl * (b_I_numerator / current_sigma_sq_e)
        posterior_mean_I_kl = posterior_mean_I_kl.item() if isinstance(posterior_mean_I_kl, np.ndarray) and posterior_mean_I_kl.ndim == 0 else float(posterior_mean_I_kl)

        if posterior_variance_I_kl < 0: 
            samplers_logger.warning(f"Negative posterior variance ({posterior_variance_I_kl:.3e}) for epistatic pair {snp_pair_indices_tuple}. Setting effect to 0.")
            new_sampled_I_kl = 0.0
            inclusion_indicator_gamma_kl = 0 
        else:
            new_sampled_I_kl = np.random.normal(posterior_mean_I_kl, np.sqrt(posterior_variance_I_kl))
    
    return new_sampled_I_kl, inclusion_indicator_gamma_kl

# --- Code from bayesadme_model.py ---
model_logger = logging.getLogger("BayesADME_Model")

class BayesADMEModel:
    """
    Implements the BayesADME model using Gibbs sampling for posterior inference.
    """
    def __init__(self, config: ModelConfig):
        self.config = config; self.data = None
        self.mu = 0.0; self.sigma_sq_e = 1.0
        self.additive_effects = None; self.additive_variances_fixed = None
        self.additive_pi_by_cat = None; self.additive_component_assignments = None
        self.dominance_effects = None; self.dominance_variances_fixed = None
        self.dominance_pi_by_cat = None; self.dominance_component_assignments = None
        self.epistatic_effects = {}; self.epistatic_gammas = {}
        self.active_epistatic_pairs = []; self.epistatic_variance_fixed = None
        self.posterior_mu_samples = []; self.posterior_sigma_sq_e_samples = []
        self.posterior_additive_effects_sum = None; self.posterior_dominance_effects_sum = None
        self.posterior_epistatic_effects_conditional_sum = defaultdict(float)
        self.posterior_epistatic_inclusion_counts = defaultdict(int)
        self.posterior_additive_pi_by_cat_samples = []; self.posterior_dominance_pi_by_cat_samples = []
        self.num_samples_collected = 0
        self.snp_to_category_idx_map = None; self.num_functional_categories = 1
        self.N_inds, self.M_snps = 0, 0
        self.posterior_mu_mean = None; self.posterior_sigma_sq_e_mean = None
        self.posterior_additive_effects_mean = None; self.posterior_dominance_effects_mean = None
        self.posterior_epistatic_effects_mean = {}; self.posterior_epistatic_pip = {}
        self.gebv = None; self.variance_components = {}

    def _initialize_parameters_from_data_and_config(self):
        model_logger.info("Initializing model parameters from data and configuration...")
        self.N_inds = self.data['N']; self.M_snps = self.data['M']
        self.mu = np.mean(self.data['phenotypes']) if self.data['phenotypes'] is not None else 0.0
        phenotypes_flat_initial = self.data['phenotypes'].flatten()
        pheno_centered_initial = phenotypes_flat_initial - self.mu
        self.sigma_sq_e = np.var(pheno_centered_initial) if self.N_inds > 1 else 1.0
        if self.sigma_sq_e <= 1e-9: self.sigma_sq_e = 1.0 
        self.snp_to_category_idx_map = self.data.get('snp_annotations_mapped', np.zeros(self.M_snps, dtype=int))
        self.num_functional_categories = self.data.get('num_functional_categories', 1)

        if self.config.get('include_additive', True):
            self.additive_effects = np.zeros(self.M_snps)
            self.additive_variances_fixed = np.array(self.config.get('prior_additive_variances_fixed'))
            num_add_mix_comp = self.config.get('additive_num_mixture_components')
            default_add_pi_for_one_category = np.ones(num_add_mix_comp) / num_add_mix_comp
            self.additive_pi_by_cat = np.tile(default_add_pi_for_one_category, (self.num_functional_categories, 1))
            self.additive_component_assignments = np.zeros(self.M_snps, dtype=int) 

        if self.config.get('include_dominance', False) and self.data.get('dominance_genotypes') is not None:
            self.dominance_effects = np.zeros(self.M_snps)
            self.dominance_variances_fixed = np.array(self.config.get('prior_dominance_variances_fixed'))
            num_dom_mix_comp = self.config.get('dominance_num_mixture_components')
            default_dom_pi_for_one_category = np.ones(num_dom_mix_comp) / num_dom_mix_comp
            self.dominance_pi_by_cat = np.tile(default_dom_pi_for_one_category, (self.num_functional_categories, 1))
            self.dominance_component_assignments = np.zeros(self.M_snps, dtype=int)
        else: 
            self.dominance_effects = np.zeros(self.M_snps)
            self.dominance_pi_by_cat = None; self.dominance_variances_fixed = None
            self.dominance_component_assignments = np.zeros(self.M_snps, dtype=int)

        if self.config.get('include_epistasis', False):
            self.epistatic_effects = {}; self.epistatic_gammas = {}
            self.active_epistatic_pairs = []; self.epistatic_variance_fixed = self.config.get('prior_epistasis_variance_fixed')
        
        self.posterior_additive_effects_sum = np.zeros(self.M_snps)
        self.posterior_dominance_effects_sum = np.zeros(self.M_snps)
        self.posterior_epistatic_effects_conditional_sum = defaultdict(float)
        self.posterior_epistatic_inclusion_counts = defaultdict(int)
        self.num_samples_collected = 0
        model_logger.info("Model parameters initialized successfully.")

    def _calculate_current_total_residuals(self, y_true_flat_vector):
        residuals = y_true_flat_vector - self.mu
        if self.config.get('include_additive'):
            residuals -= np.dot(self.data['additive_genotypes'], self.additive_effects)
        if self.config.get('include_dominance') and self.data.get('dominance_genotypes') is not None:
            residuals -= np.dot(self.data['dominance_genotypes'], self.dominance_effects)
        if self.config.get('include_epistasis') and self.active_epistatic_pairs:
            X_add_genotypes_matrix = self.data['additive_genotypes']
            for k_snp_idx, l_snp_idx in self.active_epistatic_pairs:
                if self.epistatic_gammas.get((k_snp_idx, l_snp_idx), 0) == 1:
                    current_epistatic_effect_val = self.epistatic_effects.get((k_snp_idx, l_snp_idx), 0.0)
                    if abs(current_epistatic_effect_val) > 1e-9:
                        interaction_term_w_kl_column = calculate_epistatic_interaction_term(
                            X_add_genotypes_matrix[:, k_snp_idx], X_add_genotypes_matrix[:, l_snp_idx]
                        )
                        residuals -= interaction_term_w_kl_column * current_epistatic_effect_val
        return residuals

    def run_mcmc(self, data_from_handler_dict):
        self.data = data_from_handler_dict
        self._initialize_parameters_from_data_and_config()
        y_true_flat_vector = self.data['phenotypes'].flatten()
        X_additive_genotypes = self.data['additive_genotypes'] 
        X_dominance_genotypes = self.data.get('dominance_genotypes')
        num_total_iterations = self.config.get('num_iterations')
        num_burn_in_iterations = self.config.get('burn_in')
        thinning_interval = self.config.get('thinning')
        if self.config.get('random_seed') is not None: np.random.seed(self.config.get('random_seed'))
        model_logger.info(f"Starting MCMC: {num_total_iterations} iterations, {num_burn_in_iterations} burn-in, {thinning_interval} thinning.")
        mcmc_overall_start_time = time.time()
        current_total_residuals_vector = self._calculate_current_total_residuals(y_true_flat_vector)

        for iteration_count in range(1, num_total_iterations + 1):
            iteration_loop_start_time = time.time()
            mu_old_value = self.mu
            y_residuals_for_mu_sampling = current_total_residuals_vector + mu_old_value 
            self.mu = sample_mu_overall(y_residuals_for_mu_sampling, self.sigma_sq_e, self.N_inds)
            current_total_residuals_vector -= (self.mu - mu_old_value) 

            if self.config.get('include_additive'):
                shuffled_snp_indices_additive = np.random.permutation(self.M_snps)
                counts_additive_components_this_iter = np.zeros((self.num_functional_categories, self.config.get('additive_num_mixture_components')))
                for snp_j_index in shuffled_snp_indices_additive:
                    additive_genotype_col_j = X_additive_genotypes[:, snp_j_index]
                    additive_effect_j_old = self.additive_effects[snp_j_index]
                    y_residuals_for_additive_j = current_total_residuals_vector + additive_genotype_col_j * additive_effect_j_old
                    snp_j_category_index = self.snp_to_category_idx_map[snp_j_index]
                    new_additive_effect_j, assigned_component_idx_a = sample_single_snp_effect_from_mixture(
                        y_residuals_for_additive_j, additive_genotype_col_j, self.sigma_sq_e,
                        self.additive_pi_by_cat[snp_j_category_index, :], self.additive_variances_fixed, snp_j_index
                    )
                    self.additive_effects[snp_j_index] = new_additive_effect_j
                    self.additive_component_assignments[snp_j_index] = assigned_component_idx_a
                    counts_additive_components_this_iter[snp_j_category_index, assigned_component_idx_a] += 1
                    current_total_residuals_vector -= additive_genotype_col_j * (new_additive_effect_j - additive_effect_j_old)
                self.additive_pi_by_cat = sample_component_mixture_proportions(
                    counts_additive_components_this_iter, self.config.get('prior_additive_pi_alpha'), self.num_functional_categories
                )

            if self.config.get('include_dominance') and X_dominance_genotypes is not None:
                shuffled_snp_indices_dominance = np.random.permutation(self.M_snps) 
                counts_dominance_components_this_iter = np.zeros((self.num_functional_categories, self.config.get('dominance_num_mixture_components')))
                for snp_j_index in shuffled_snp_indices_dominance:
                    dominance_genotype_col_j = X_dominance_genotypes[:, snp_j_index]
                    dominance_effect_j_old = self.dominance_effects[snp_j_index]
                    y_residuals_for_dominance_j = current_total_residuals_vector + dominance_genotype_col_j * dominance_effect_j_old
                    snp_j_category_index = self.snp_to_category_idx_map[snp_j_index]
                    new_dominance_effect_j, assigned_component_idx_d = sample_single_snp_effect_from_mixture(
                        y_residuals_for_dominance_j, dominance_genotype_col_j, self.sigma_sq_e,
                        self.dominance_pi_by_cat[snp_j_category_index, :], self.dominance_variances_fixed, snp_j_index
                    )
                    self.dominance_effects[snp_j_index] = new_dominance_effect_j
                    self.dominance_component_assignments[snp_j_index] = assigned_component_idx_d
                    counts_dominance_components_this_iter[snp_j_category_index, assigned_component_idx_d] += 1
                    current_total_residuals_vector -= dominance_genotype_col_j * (new_dominance_effect_j - dominance_effect_j_old)
                self.dominance_pi_by_cat = sample_component_mixture_proportions(
                    counts_dominance_components_this_iter, self.config.get('prior_dominance_pi_alpha'), self.num_functional_categories
                )

            if self.config.get('include_epistasis'):
                pairs_to_remove_from_active_list = []
                for k_snp_idx, l_snp_idx in list(self.active_epistatic_pairs): 
                    interaction_term_w_kl_column = calculate_epistatic_interaction_term(X_additive_genotypes[:, k_snp_idx], X_additive_genotypes[:, l_snp_idx])
                    epistatic_effect_kl_old = self.epistatic_effects.get((k_snp_idx, l_snp_idx), 0.0)
                    y_residuals_for_epistatic_kl = current_total_residuals_vector + interaction_term_w_kl_column * epistatic_effect_kl_old
                    new_epistatic_effect_kl, new_gamma_kl = sample_single_epistatic_effect(
                        y_residuals_for_epistatic_kl, interaction_term_w_kl_column, self.sigma_sq_e,
                        self.config.get('prior_epistasis_pi0'), self.epistatic_variance_fixed, (k_snp_idx, l_snp_idx)
                    )
                    self.epistatic_effects[(k_snp_idx, l_snp_idx)] = new_epistatic_effect_kl
                    self.epistatic_gammas[(k_snp_idx, l_snp_idx)] = new_gamma_kl
                    current_total_residuals_vector -= interaction_term_w_kl_column * (new_epistatic_effect_kl - epistatic_effect_kl_old)
                    if new_gamma_kl == 0 and (k_snp_idx,l_snp_idx) in self.active_epistatic_pairs: 
                        pairs_to_remove_from_active_list.append((k_snp_idx, l_snp_idx))
                for pair_to_remove in pairs_to_remove_from_active_list: self.active_epistatic_pairs.remove(pair_to_remove)
                max_active_pairs_allowed = self.config.get('epistasis_max_active_pairs')
                num_new_pairs_to_propose = self.config.get('epistasis_snp_pair_proposal_count', self.M_snps // 10)
                for _ in range(num_new_pairs_to_propose):
                    if len(self.active_epistatic_pairs) >= max_active_pairs_allowed: break 
                    proposed_k_idx = np.random.randint(self.M_snps); proposed_l_idx = np.random.randint(self.M_snps)
                    if proposed_k_idx == proposed_l_idx: continue
                    current_pair_key = tuple(sorted((proposed_k_idx, proposed_l_idx)))
                    if current_pair_key in self.active_epistatic_pairs or self.epistatic_gammas.get(current_pair_key,0) == 1 : continue 
                    interaction_term_w_prop_kl_col = calculate_epistatic_interaction_term(X_additive_genotypes[:, current_pair_key[0]], X_additive_genotypes[:, current_pair_key[1]])
                    epistatic_effect_prop_kl_old = self.epistatic_effects.get(current_pair_key, 0.0) 
                    y_residuals_for_proposed_kl = current_total_residuals_vector + interaction_term_w_prop_kl_col * epistatic_effect_prop_kl_old
                    new_epistatic_effect_prop, new_gamma_prop = sample_single_epistatic_effect(
                        y_residuals_for_proposed_kl, interaction_term_w_prop_kl_col, self.sigma_sq_e,
                        self.config.get('prior_epistasis_pi0'), self.epistatic_variance_fixed, current_pair_key
                    )
                    if new_gamma_prop == 1: 
                        self.epistatic_effects[current_pair_key] = new_epistatic_effect_prop
                        self.epistatic_gammas[current_pair_key] = 1
                        if current_pair_key not in self.active_epistatic_pairs: self.active_epistatic_pairs.append(current_pair_key)
                        current_total_residuals_vector -= interaction_term_w_prop_kl_col * (new_epistatic_effect_prop - epistatic_effect_prop_kl_old)

            self.sigma_sq_e = sample_residual_variance(current_total_residuals_vector, self.N_inds,
                nu0=self.config.get('prior_sigma_e_nu0', 0.001), s0=self.config.get('prior_sigma_e_s0', 0.001)
            )

            if iteration_count > num_burn_in_iterations and (iteration_count - num_burn_in_iterations) % thinning_interval == 0:
                self.posterior_mu_samples.append(self.mu); self.posterior_sigma_sq_e_samples.append(self.sigma_sq_e)
                if self.config.get('include_additive'): self.posterior_additive_effects_sum += self.additive_effects
                if self.config.get('include_dominance') and X_dominance_genotypes is not None: self.posterior_dominance_effects_sum += self.dominance_effects
                if self.config.get('include_epistasis'):
                    for pair_key, gamma_value in self.epistatic_gammas.items(): 
                        if gamma_value == 1: 
                             self.posterior_epistatic_inclusion_counts[pair_key] += 1
                             self.posterior_epistatic_effects_conditional_sum[pair_key] += self.epistatic_effects.get(pair_key,0.0)
                if self.additive_pi_by_cat is not None: self.posterior_additive_pi_by_cat_samples.append(self.additive_pi_by_cat.copy())
                if self.dominance_pi_by_cat is not None: self.posterior_dominance_pi_by_cat_samples.append(self.dominance_pi_by_cat.copy())
                self.num_samples_collected += 1

            if iteration_count % (max(1, num_total_iterations // 20)) == 0 or iteration_count == 1: 
                 iteration_loop_time_taken = time.time() - iteration_loop_start_time
                 model_logger.info(f"Iter {iteration_count}/{num_total_iterations}. mu={self.mu:.3f}, sig_e2={self.sigma_sq_e:.3f}. ActiveEpi={len(self.active_epistatic_pairs)}. Time/iter: {iteration_loop_time_taken:.3f}s.")
        
        total_mcmc_duration = time.time() - mcmc_overall_start_time
        model_logger.info(f"MCMC finished in {total_mcmc_duration:.2f} seconds.")
        self._process_and_summarize_posterior_samples()

    def _process_and_summarize_posterior_samples(self):
        if self.num_samples_collected == 0:
            model_logger.warning("No samples collected. Summaries based on last state or zero.")
            self.posterior_mu_mean = self.mu; self.posterior_sigma_sq_e_mean = self.sigma_sq_e
            self.posterior_additive_effects_mean = self.additive_effects if self.additive_effects is not None else np.zeros(self.M_snps)
            self.posterior_dominance_effects_mean = self.dominance_effects if self.dominance_effects is not None else np.zeros(self.M_snps)
            if self.config.get('include_epistasis'):
                for pair_key, gamma_val_last_iter in self.epistatic_gammas.items():
                    self.posterior_epistatic_pip[pair_key] = float(gamma_val_last_iter)
                    self.posterior_epistatic_effects_mean[pair_key] = self.epistatic_effects.get(pair_key,0.0) * float(gamma_val_last_iter)
        else: 
            self.posterior_mu_mean = np.mean(self.posterior_mu_samples)
            self.posterior_sigma_sq_e_mean = np.mean(self.posterior_sigma_sq_e_samples)
            if self.config.get('include_additive'): self.posterior_additive_effects_mean = self.posterior_additive_effects_sum / self.num_samples_collected
            else: self.posterior_additive_effects_mean = np.zeros(self.M_snps)
            if self.config.get('include_dominance') and self.data.get('dominance_genotypes') is not None:
                self.posterior_dominance_effects_mean = self.posterior_dominance_effects_sum / self.num_samples_collected
            else: self.posterior_dominance_effects_mean = np.zeros(self.M_snps)
            if self.config.get('include_epistasis'):
                for pair_key, count_included_in_model in self.posterior_epistatic_inclusion_counts.items():
                    self.posterior_epistatic_pip[pair_key] = count_included_in_model / self.num_samples_collected
                    self.posterior_epistatic_effects_mean[pair_key] = self.posterior_epistatic_effects_conditional_sum.get(pair_key,0.0) / self.num_samples_collected
        
        self.gebv = np.full(self.N_inds, self.posterior_mu_mean)
        if self.config.get('include_additive'): self.gebv += np.dot(self.data['additive_genotypes'], self.posterior_additive_effects_mean)
        if self.config.get('include_dominance') and self.data.get('dominance_genotypes') is not None:
             self.gebv += np.dot(self.data['dominance_genotypes'], self.posterior_dominance_effects_mean)
        if self.config.get('include_epistasis'):
            X_additive_genotypes_matrix = self.data['additive_genotypes']
            for pair_key, mean_epistatic_effect_overall in self.posterior_epistatic_effects_mean.items():
                if abs(mean_epistatic_effect_overall) > 1e-9 : 
                    k_snp_idx, l_snp_idx = pair_key
                    interaction_term_w_kl_column = calculate_epistatic_interaction_term(X_additive_genotypes_matrix[:, k_snp_idx], X_additive_genotypes_matrix[:, l_snp_idx])
                    self.gebv += interaction_term_w_kl_column * mean_epistatic_effect_overall
        self._estimate_variance_components()
        model_logger.info("Posterior samples processed and summarized successfully.")

    def _estimate_variance_components(self):
        self.variance_components['Ve'] = self.posterior_sigma_sq_e_mean
        Va, Vd, Vaa = 0.0, 0.0, 0.0
        if self.config.get('include_additive'):
            variances_of_X_additive_columns = np.var(self.data['additive_genotypes'], axis=0)
            Va = np.sum(variances_of_X_additive_columns * (self.posterior_additive_effects_mean**2))
        self.variance_components['Va'] = Va
        if self.config.get('include_dominance') and self.data.get('dominance_genotypes') is not None:
            variances_of_X_dominance_columns = np.var(self.data['dominance_genotypes'], axis=0) 
            Vd = np.sum(variances_of_X_dominance_columns * (self.posterior_dominance_effects_mean**2))
        self.variance_components['Vd'] = Vd
        if self.config.get('include_epistasis') and self.posterior_epistatic_effects_mean:
            X_additive_genotypes_matrix = self.data['additive_genotypes']
            current_Vaa_sum = 0.0
            for pair_key, mean_overall_epistatic_effect in self.posterior_epistatic_effects_mean.items():
                 if abs(mean_overall_epistatic_effect) > 1e-9 :
                    k_snp_idx, l_snp_idx = pair_key
                    interaction_term_w_kl_column = calculate_epistatic_interaction_term(X_additive_genotypes_matrix[:, k_snp_idx], X_additive_genotypes_matrix[:, l_snp_idx])
                    variance_of_w_kl_column = np.var(interaction_term_w_kl_column)
                    current_Vaa_sum += variance_of_w_kl_column * (mean_overall_epistatic_effect**2)
            Vaa = current_Vaa_sum
        self.variance_components['Vaa'] = Vaa
        total_genetic_variance_est = Va + Vd + Vaa
        total_phenotypic_variance_est = total_genetic_variance_est + self.variance_components['Ve']
        self.variance_components['V_genetic_total_est'] = total_genetic_variance_est
        self.variance_components['Vp_est'] = total_phenotypic_variance_est
        model_logger.info(f"Estimated Variance Components: VA={Va:.4f}, VD={Vd:.4f}, VAA={Vaa:.4f}, Ve={self.variance_components['Ve']:.4f}, VP_est={total_phenotypic_variance_est:.4f}")

    def save_results(self):
        output_directory = self.config.get('output_dir'); file_prefix = self.config.get('output_prefix')
        os.makedirs(output_directory, exist_ok=True)
        summary_file_path = os.path.join(output_directory, f"{file_prefix}_summary.txt")
        with open(summary_file_path, 'w') as f_summary:
            f_summary.write(f"BayesADME Run Summary\n{'-'*24}\nConfiguration Parameters:\n")
            for key_conf, value_conf in sorted(self.config.config.items()): f_summary.write(f"  {key_conf}: {value_conf}\n")
            f_summary.write(f"{'-'*24}\nPosterior Means & Variance Component Summaries:\n")
            f_summary.write(f"  Posterior Mean mu: {self.posterior_mu_mean:.4f}\n")
            f_summary.write(f"  Posterior Mean sigma_sq_e (Ve): {self.variance_components.get('Ve', 0.0):.4f}\n")
            f_summary.write(f"  Estimated VA: {self.variance_components.get('Va', 0.0):.4f}\n")
            f_summary.write(f"  Estimated VD: {self.variance_components.get('Vd', 0.0):.4f}\n")
            f_summary.write(f"  Estimated VAA: {self.variance_components.get('Vaa', 0.0):.4f}\n")
            f_summary.write(f"  Estimated Total Genetic Variance: {self.variance_components.get('V_genetic_total_est', 0.0):.4f}\n")
            f_summary.write(f"  Estimated Total Phenotypic Variance (VP_est): {self.variance_components.get('Vp_est', 0.0):.4f}\n")
            f_summary.write(f"{'-'*24}\nMCMC Samples Collected: {self.num_samples_collected}\n")

        final_aligned_individual_ids = self.data.get('aligned_individual_ids', [f"Individual_{i+1}" for i in range(self.N_inds)])
        gebv_df = pd.DataFrame({'IndividualID': final_aligned_individual_ids, 'GEBV_or_TotalGeneticValue': self.gebv.flatten()})
        gebv_df.to_csv(os.path.join(output_directory, f"{file_prefix}_gebv.csv"), index=False, float_format='%.5f')

        aligned_snp_ids_list = self.data.get('aligned_snp_ids', [f"SNP_{i+1}" for i in range(self.M_snps)])
        if self.config.get('include_additive'):
            additive_effects_df = pd.DataFrame({'SNP_ID': aligned_snp_ids_list, 'AdditiveEffectMean': self.posterior_additive_effects_mean})
            additive_effects_df.to_csv(os.path.join(output_directory, f"{file_prefix}_additive_effects.csv"), index=False, float_format='%.5f')
        if self.config.get('include_dominance') and self.data.get('dominance_genotypes') is not None:
            dominance_effects_df = pd.DataFrame({'SNP_ID': aligned_snp_ids_list, 'DominanceEffectMean': self.posterior_dominance_effects_mean})
            dominance_effects_df.to_csv(os.path.join(output_directory, f"{file_prefix}_dominance_effects.csv"), index=False, float_format='%.5f')
        if self.config.get('include_epistasis') and self.posterior_epistatic_effects_mean:
            epistatic_results_data = []
            for (k_idx, l_idx), mean_overall_effect in self.posterior_epistatic_effects_mean.items():
                pip_for_pair = self.posterior_epistatic_pip.get((k_idx, l_idx), 0.0)
                if pip_for_pair > 1e-4 or abs(mean_overall_effect) > 1e-6 : 
                    epistatic_results_data.append({'SNP1_ID': aligned_snp_ids_list[k_idx], 'SNP2_ID': aligned_snp_ids_list[l_idx],
                                                   'EpistaticEffectMean_Overall': mean_overall_effect, 'PIP': pip_for_pair})
            if epistatic_results_data:
                epistatic_effects_df = pd.DataFrame(epistatic_results_data).sort_values(by='PIP', ascending=False)
                epistatic_effects_df.to_csv(os.path.join(output_directory, f"{file_prefix}_epistatic_effects.csv"), index=False, float_format='%.5f')
        if self.num_samples_collected > 0 :
            mcmc_trace_df = pd.DataFrame({'SampledIteration': range(1, self.num_samples_collected + 1), 'mu': self.posterior_mu_samples, 'sigma_sq_e': self.posterior_sigma_sq_e_samples})
            mcmc_trace_df.to_csv(os.path.join(output_directory, f"{file_prefix}_mcmc_trace_hyperparams.csv"), index=False, float_format='%.5f')
        model_logger.info(f"All results saved to directory: {output_directory}")

# --- Code from cli.py ---
cli_logger = logging.getLogger("BayesADME_CLI_Main") 

def execute_run_analysis_pipeline(cli_args):
    cli_logger.info("Initializing BayesADME analysis run via Command Line Interface...")
    overall_run_start_time = time.time()
    try:
        model_config_instance = ModelConfig(config_file_path=cli_args.config)
        cli_parameter_overrides = {
            "num_iterations": cli_args.num_iterations, "burn_in": cli_args.burn_in, "thinning": cli_args.thinning,
            "output_dir": cli_args.output_dir, "output_prefix": cli_args.output_prefix,
            "random_seed": cli_args.random_seed, "phenotype_file": cli_args.phenotype_file,
            "additive_genotype_file": cli_args.additive_genotype_file,
            "dominance_genotype_file": cli_args.dominance_genotype_file,
            "annotation_file": cli_args.annotation_file
        }
        for config_key, cli_value in cli_parameter_overrides.items():
            if cli_value is not None: 
                cli_logger.info(f"Overriding config '{config_key}' with CLI value: {cli_value}")
                model_config_instance.config[config_key] = cli_value 
        model_config_instance._validate_config() 
        model_config_instance.log_config() 
        cli_logger.info("Initializing DataHandler for data loading and preprocessing...")
        data_handler_instance = DataHandler(model_config_instance)
        processed_data_for_model = data_handler_instance.load_and_process_data()
        cli_logger.info("Initializing BayesADMEModel...")
        bayesadme_model_instance = BayesADMEModel(model_config_instance)
        cli_logger.info("Starting MCMC sampling procedure...")
        bayesadme_model_instance.run_mcmc(processed_data_for_model) 
        cli_logger.info("MCMC finished. Saving results...")
        bayesadme_model_instance.save_results()
        overall_run_duration = time.time() - overall_run_start_time
        cli_logger.info(f"BayesADME analysis pipeline completed successfully in {overall_run_duration:.2f} seconds.")
    except FileNotFoundError as e_fnf:
        cli_logger.error(f"File Not Found Error: {e_fnf}. Check config and paths.")
        sys.exit(1) 
    except ValueError as e_val:
        cli_logger.error(f"Value Error: {e_val}. Check parameters and data consistency.")
        sys.exit(1)
    except Exception as e_unexpected:
        cli_logger.error(f"An unexpected error occurred during the BayesADME run: {e_unexpected}", exc_info=True)
        sys.exit(1)

def main_cli_entrypoint():
    parser = argparse.ArgumentParser(
        description="BayesADME: Bayesian Genomic Prediction Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose (DEBUG level) logging.")
    subparsers = parser.add_subparsers(dest="command", title="Available Commands", help="Run 'bayesadme <command> --help' for command help.")
    subparsers.required = True
    parser_run = subparsers.add_parser("run", help="Run the BayesADME MCMC analysis pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_run.add_argument("--config", type=str, required=True, help="Path to the main YAML configuration file.")
    parser_run.add_argument("--num_iterations", type=int, help="Override: Total MCMC iterations.")
    parser_run.add_argument("--burn_in", type=int, help="Override: MCMC burn-in period.")
    parser_run.add_argument("--thinning", type=int, help="Override: MCMC thinning interval.")
    parser_run.add_argument("--output_dir", type=str, help="Override: Output directory path.")
    parser_run.add_argument("--output_prefix", type=str, help="Override: Prefix for output files.")
    parser_run.add_argument("--random_seed", type=int, help="Override: Random seed.")
    parser_run.add_argument("--phenotype_file", type=str, help="Override: Phenotype file path.")
    parser_run.add_argument("--additive_genotype_file", type=str, help="Override: Additive genotype file path.")
    parser_run.add_argument("--dominance_genotype_file", type=str, help="Override: Dominance genotype file path.")
    parser_run.add_argument("--annotation_file", type=str, help="Override: SNP annotation file path.")
    parser_run.set_defaults(func=execute_run_analysis_pipeline) 
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) 
        for handler in logging.getLogger().handlers: handler.setLevel(logging.DEBUG)
        cli_logger.info("Verbose logging enabled (DEBUG level).")
    if hasattr(args, 'func'): args.func(args)
    else: parser.print_help()

if __name__ == "__main__":
    # This makes the script runnable directly for CLI interaction.
    # Example: python your_combined_script_name.py run --config path/to/config.yml
    
    # Setup basic logging if run as a script and not already configured by a library import.
    # This ensures that log messages from all modules are visible.
    if not logging.getLogger().hasHandlers(): # Check if root logger already has handlers
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Example Test Code (originally from individual module tests) ---
    # This section can be used for basic testing of the combined script.
    # It's good practice to have a more structured testing suite for a real package.
    
    # Example test for ModelConfig (files don't need to exist for this part of config test)
    # config_logger.info("--- Testing ModelConfig within combined script ---")
    # try:
    #     test_mc_dict = {"phenotype_file": "test_p.txt", "additive_genotype_file": "test_a.txt", "num_iterations": 5}
    #     test_mc = ModelConfig(config_dict=test_mc_dict)
    #     test_mc.log_config()
    # except Exception as e_mc_test:
    #     config_logger.error(f"Error in combined ModelConfig test: {e_mc_test}")

    # Example test for DataHandler (requires dummy files)
    # To run this, you would need to create the dummy files as in data_handler.py's original test.
    # data_handler_logger.info("--- Testing DataHandler within combined script (requires dummy files) ---")
    # try:
    #     # Create dummy files and config for DataHandler test here if needed
    #     # ... (setup code similar to data_handler.py's if __name__ == '__main__') ...
    #     # config_dh_test = ModelConfig(config_dict={...paths to dummy files...})
    #     # dh_test = DataHandler(config_dh_test)
    #     # processed_data_dh = dh_test.load_and_process_data()
    #     # data_handler_logger.info(f"DataHandler test processed N={processed_data_dh['N']}, M={processed_data_dh['M']}")
    #     pass # Placeholder for actual test setup
    # except Exception as e_dh_test_combined:
    #     data_handler_logger.error(f"Error in combined DataHandler test: {e_dh_test_combined}")

    # The main entry point for CLI execution:
    main_cli_entrypoint()
