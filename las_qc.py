"""
LAS File Quality Control Module
Handles LAS file validation, analysis, and interpretation
"""
import lasio
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings

warnings.filterwarnings('ignore')


class LASQCAnalyzer:
    """Comprehensive LAS file quality control and analysis"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.las = None
        self.df = None
        self.well_info = {}
        self.qc_results = {}
        self.curves_info = {}

    def load_las(self) -> bool:
        """Load LAS file and convert to DataFrame"""
        try:
            self.las = lasio.read(self.filepath)
            self.df = self.las.df().reset_index()

            # Extract well information
            self.well_info = {
                'well_name': self._get_well_header('WELL', 'Unknown'),
                'start_depth': float(self._get_well_header('STRT', 0)),
                'stop_depth': float(self._get_well_header('STOP', 0)),
                'step': float(self._get_well_header('STEP', 0)),
                'company': self._get_well_header('COMP', 'Unknown'),
                'field': self._get_well_header('FLD', 'Unknown'),
                'location': self._get_well_header('LOC', 'Unknown'),
                'province': self._get_well_header('PROV', 'Unknown'),
                'service_company': self._get_well_header('SRVC', 'Unknown'),
                'date': self._get_well_header('DATE', 'Unknown'),
                'api': self._get_well_header('API', 'Unknown'),
                'uwi': self._get_well_header('UWI', 'Unknown'),
            }

            # Extract curve information
            for curve in self.las.curves:
                self.curves_info[curve.mnemonic] = {
                    'unit': curve.unit if curve.unit else 'Unknown',
                    'description': curve.descr if curve.descr else 'No description',
                    'min': float(self.df[curve.mnemonic].min()) if curve.mnemonic in self.df.columns else None,
                    'max': float(self.df[curve.mnemonic].max()) if curve.mnemonic in self.df.columns else None,
                    'null_count': int(self.df[curve.mnemonic].isna().sum()) if curve.mnemonic in self.df.columns else None,
                }

            return True
        except Exception as e:
            self.qc_results['load_error'] = str(e)
            return False

    def _get_well_header(self, key: str, default: str) -> str:
        """Safely get well header value"""
        try:
            value = self.las.well[key].value
            return str(value) if value else default
        except:
            return default

    def run_quality_control(self) -> Dict:
        """Run comprehensive QC checks"""
        if self.df is None:
            return {'error': 'No LAS file loaded'}

        qc = {
            'file_info': {
                'total_curves': len(self.las.curves),
                'total_points': len(self.df),
                'depth_range': {
                    'min': float(self.df.iloc[:, 0].min()),
                    'max': float(self.df.iloc[:, 0].max()),
                },
                'sample_rate': self._detect_sample_rate(),
            },
            'data_gaps': self._check_data_gaps(),
            'null_values': self._check_null_values(),
            'outliers': self._detect_outliers(),
            'curve_statistics': self._compute_statistics(),
            'duplicates': self._check_duplicates(),
            'data_quality_score': 0,
            'issues': [],
            'warnings': [],
        }

        # Calculate quality score and collect issues
        qc['data_quality_score'] = self._calculate_quality_score(qc)

        self.qc_results = qc
        return qc

    def _detect_sample_rate(self) -> str:
        """Detect sampling rate of the data"""
        depth_col = self.df.columns[0]
        diffs = self.df[depth_col].diff().dropna()
        if len(diffs) > 0:
            mode_diff = diffs.mode()
            if len(mode_diff) > 0:
                return f"{mode_diff.iloc[0]:.4f}"
        return "Variable"

    def _check_data_gaps(self) -> Dict:
        """Check for data gaps in depth"""
        depth_col = self.df.columns[0]
        depth = self.df[depth_col]

        # Detect gaps larger than 2x the typical step
        diffs = depth.diff().dropna()
        if len(diffs) > 0:
            typical_step = diffs.median()
            threshold = typical_step * 2.5
            gaps = diffs[diffs > threshold]

            return {
                'has_gaps': len(gaps) > 0,
                'gap_count': len(gaps),
                'largest_gap': float(gaps.max()) if len(gaps) > 0 else 0,
                'gap_locations': [
                    {'depth': float(depth.iloc[i]), 'size': float(gaps.iloc[idx])}
                    for idx, i in enumerate(gaps.index[:10])  # Limit to first 10
                ],
            }
        return {'has_gaps': False, 'gap_count': 0}

    def _check_null_values(self) -> Dict:
        """Check for null/missing values in each curve"""
        null_analysis = {}
        total_rows = len(self.df)

        for col in self.df.columns[1:]:  # Skip depth column
            null_count = self.df[col].isna().sum()
            null_percent = (null_count / total_rows) * 100
            null_analysis[col] = {
                'count': int(null_count),
                'percent': float(null_percent),
                'status': 'critical' if null_percent > 50 else 'warning' if null_percent > 20 else 'good',
            }

        return null_analysis

    def _detect_outliers(self) -> Dict:
        """Detect outliers using IQR method"""
        outliers = {}

        for col in self.df.columns[1:]:
            data = self.df[col].dropna()
            if len(data) > 10:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_data = data[(data < lower_bound) | (data > upper_bound)]

                outliers[col] = {
                    'count': len(outlier_data),
                    'percent': float(len(outlier_data) / len(data) * 100),
                    'min_val': float(outlier_data.min()) if len(outlier_data) > 0 else None,
                    'max_val': float(outlier_data.max()) if len(outlier_data) > 0 else None,
                }

        return outliers

    def _compute_statistics(self) -> Dict:
        """Compute statistical summary for each curve"""
        stats_summary = {}

        for col in self.df.columns[1:]:
            data = self.df[col].dropna()
            if len(data) > 0:
                stats_summary[col] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'p10': float(data.quantile(0.10)),
                    'p50': float(data.quantile(0.50)),
                    'p90': float(data.quantile(0.90)),
                    'skewness': float(stats.skew(data)) if len(data) > 2 else 0,
                    'kurtosis': float(stats.kurtosis(data)) if len(data) > 2 else 0,
                }

        return stats_summary

    def _check_duplicates(self) -> Dict:
        """Check for duplicate depth values"""
        depth_col = self.df.columns[0]
        duplicates = self.df[depth_col].duplicated().sum()

        return {
            'has_duplicates': duplicates > 0,
            'duplicate_count': int(duplicates),
        }

    def _calculate_quality_score(self, qc: Dict) -> float:
        """Calculate overall data quality score"""
        score = 100.0

        # Penalty for data gaps
        if qc['data_gaps']['has_gaps']:
            score -= min(20, qc['data_gaps']['gap_count'] * 2)

        # Penalty for null values
        for curve, info in qc['null_values'].items():
            if info['status'] == 'critical':
                score -= 5
            elif info['status'] == 'warning':
                score -= 2

        # Penalty for outliers
        for curve, info in qc['outliers'].items():
            if info['percent'] > 10:
                score -= 3
            elif info['percent'] > 5:
                score -= 1

        # Penalty for duplicates
        if qc['duplicates']['has_duplicates']:
            score -= min(10, qc['duplicates']['duplicate_count'])

        return max(0, score)

    def get_curve_aliases(self) -> Dict:
        """Map common curve aliases to standard names"""
        aliases = {
            'gamma_ray': ['GR', 'GRD', 'GRS', 'GRR', 'GRC', 'GR_CORR', 'GAMMA', 'GR_EDTC'],
            'deep_resistivity': ['RT', 'RD', 'RDEEP', 'RILD', 'RLLD', 'AT90', 'AHT90', 'RT_HRLT', 'RLA5'],
            'shallow_resistivity': ['RS', 'RSHAL', 'RILS', 'RLLS', 'AT10', 'AHT10', 'RSFL'],
            'medium_resistivity': ['RM', 'RMED', 'RILM', 'RLM', 'AT20', 'AHT20'],
            'density': ['RHOB', 'ZDEN', 'DENS', 'DEN', 'ZDENS', 'RHOZ'],
            'neutron': ['NPHI', 'CNCF', 'NEU', 'NEUT', 'TNPH', 'NPOR'],
            'sonic': ['DT', 'DTC', 'DTCO', 'AC', 'ACCO', 'SL'],
            'caliper': ['CALI', 'CAL', 'CALS', 'CALX', 'HCAL', 'LCAL'],
            'sp': ['SP', 'SPT', 'SPON', 'SPONT'],
            'pe': ['PE', 'PEF', 'PEFZ', 'PEU', 'PHIE'],
            'photoelectric': ['PE', 'PEF', 'PEFZ', 'PEU'],
        }

        detected = {}
        available_curves = [c.upper() for c in self.df.columns]

        for std_name, alias_list in aliases.items():
            for alias in alias_list:
                if alias in available_curves:
                    actual_col = [c for c in self.df.columns if c.upper() == alias][0]
                    detected[std_name] = actual_col
                    break

        return detected

    def interpret_lithology(self) -> pd.DataFrame:
        """Basic lithology interpretation using GR, RT, RHOB, NPHI"""
        curves = self.get_curve_aliases()
        df = self.df.copy()

        # Initialize interpretation columns
        df['LITHOLOGY'] = 'Unknown'
        df['POROSITY_EST'] = np.nan
        df['SW_EST'] = np.nan
        df['VSHALE'] = np.nan

        try:
            # Calculate VSHALE from GR
            if 'gamma_ray' in curves:
                gr = df[curves['gamma_ray']]
                gr_clean = 30  # Typical clean sand GR
                gr_shale = 150  # Typical shale GR
                vsh = (gr - gr_clean) / (gr_shale - gr_clean)
                df['VSHALE'] = vsh.clip(0, 1)

            # Estimate porosity from density
            if 'density' in curves:
                rhob = df[curves['density']]
                rho_matrix = 2.65  # Sandstone
                rho_fluid = 1.0  # Water
                phi_density = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
                df['POROSITY_EST'] = phi_density.clip(0, 0.4)

            # Simple lithology classification
            if 'gamma_ray' in curves and 'density' in curves:
                gr = df[curves['gamma_ray']]
                vsh = df['VSHALE'].fillna(0.3)

                conditions = [
                    (vsh < 0.3) & (gr < 80),
                    (vsh >= 0.3) & (vsh < 0.6),
                    (vsh >= 0.6),
                ]
                choices = ['Sandstone', 'Silty Sand', 'Shale']
                df['LITHOLOGY'] = np.select(conditions, choices, default='Unknown')

            # Estimate water saturation (Archie)
            if 'deep_resistivity' in curves and 'POROSITY_EST' in df.columns:
                rt = df[curves['deep_resistivity']]
                phi = df['POROSITY_EST'].fillna(0.2)
                rw = 0.1  # Assumed formation water resistivity
                a, m, n = 1, 2, 2  # Archie parameters

                sw = ((a * rw) / (phi**m * rt))**(1/n)
                df['SW_EST'] = sw.clip(0.01, 1)

        except Exception as e:
            print(f"Interpretation error: {e}")

        return df

    def get_crossplot_data(self, x_curve: str, y_curve: str, z_curve: Optional[str] = None) -> Dict:
        """Prepare data for crossplot visualization"""
        if x_curve not in self.df.columns or y_curve not in self.df.columns:
            return {'error': 'Curve not found'}

        data = {
            'x': self.df[x_curve].tolist(),
            'y': self.df[y_curve].tolist(),
            'x_name': x_curve,
            'y_name': y_curve,
            'x_unit': self.curves_info.get(x_curve, {}).get('unit', ''),
            'y_unit': self.curves_info.get(y_curve, {}).get('unit', ''),
        }

        if z_curve and z_curve in self.df.columns:
            data['z'] = self.df[z_curve].tolist()
            data['z_name'] = z_curve
            data['z_unit'] = self.curves_info.get(z_curve, {}).get('unit', '')

        # Add depth for hover info
        depth_col = self.df.columns[0]
        data['depth'] = self.df[depth_col].tolist()

        return data

    def get_log_plot_data(self, curves: List[str], depth_range: Optional[Tuple] = None) -> Dict:
        """Prepare data for log tracks"""
        depth_col = self.df.columns[0]
        df_filtered = self.df.copy()

        if depth_range:
            df_filtered = df_filtered[
                (df_filtered[depth_col] >= depth_range[0]) &
                (df_filtered[depth_col] <= depth_range[1])
            ]

        plot_data = {
            'depth': df_filtered[depth_col].tolist(),
            'depth_unit': self.curves_info.get(depth_col, {}).get('unit', ''),
            'tracks': []
        }

        for curve in curves:
            if curve in df_filtered.columns:
                plot_data['tracks'].append({
                    'name': curve,
                    'values': df_filtered[curve].tolist(),
                    'unit': self.curves_info.get(curve, {}).get('unit', ''),
                    'min': float(df_filtered[curve].min()) if not df_filtered[curve].isna().all() else 0,
                    'max': float(df_filtered[curve].max()) if not df_filtered[curve].isna().all() else 100,
                })

        return plot_data


class CurveStandards:
    """Standard curve names and expected ranges for QC"""

    EXPECTED_RANGES = {
        'GR': {'min': 0, 'max': 300, 'unit': 'API'},
        'RHOB': {'min': 1.5, 'max': 3.0, 'unit': 'g/cm3'},
        'NPHI': {'min': -0.1, 'max': 0.6, 'unit': 'v/v'},
        'RT': {'min': 0.1, 'max': 10000, 'unit': 'ohm.m'},
        'DT': {'min': 40, 'max': 150, 'unit': 'us/ft'},
        'CALI': {'min': 6, 'max': 24, 'unit': 'in'},
        'SP': {'min': -200, 'max': 200, 'unit': 'mV'},
        'PE': {'min': 0, 'max': 20, 'unit': 'barns/e'},
    }

    @classmethod
    def validate_curve_values(cls, curve_name: str, values: pd.Series) -> Dict:
        """Validate curve values against expected ranges"""
        upper_name = curve_name.upper()

        # Find matching standard
        std_name = None
        for std in cls.EXPECTED_RANGES.keys():
            if std in upper_name:
                std_name = std
                break

        if not std_name:
            return {'status': 'unknown', 'out_of_range': []}

        expected = cls.EXPECTED_RANGES[std_name]
        out_of_range = values[(values < expected['min']) | (values > expected['max'])]

        return {
            'status': 'warning' if len(out_of_range) > 0 else 'good',
            'expected_range': expected,
            'out_of_range_count': len(out_of_range),
            'out_of_range_percent': len(out_of_range) / len(values) * 100 if len(values) > 0 else 0,
        }


def detect_gas_effect(df: pd.DataFrame, gr_col: str, rt_col: str, rhob_col: str, nphi_col: str) -> pd.Series:
    """Detect potential gas effect using neutron-density crossover"""
    gas_flag = pd.Series(index=df.index, data=False)

    try:
        # Gas effect: NPHI < RHOB (in sandstone units)
        nphi = df[nphi_col]
        rhob = df[rhob_col]

        # Convert density to porosity units approximately
        rhob_porosity = (2.65 - rhob) / (2.65 - 1.0)

        # Crossover indicates gas
        crossover = nphi < rhob_porosity - 0.05  # 5% threshold
        gas_flag = crossover

    except Exception:
        pass

    return gas_flag


def calculate_net_to_gross(df: pd.DataFrame, vshale_col: str, porosity_col: str,
                           vsh_cutoff: float = 0.3, phi_cutoff: float = 0.1) -> Dict:
    """Calculate net to gross ratio"""
    try:
        vsh = df[vshale_col]
        phi = df[porosity_col]

        gross_thickness = len(df)
        net_mask = (vsh <= vsh_cutoff) & (phi >= phi_cutoff)
        net_thickness = net_mask.sum()

        return {
            'gross_thickness': gross_thickness,
            'net_thickness': int(net_thickness),
            'net_to_gross': float(net_thickness / gross_thickness) if gross_thickness > 0 else 0,
            'vsh_cutoff': vsh_cutoff,
            'phi_cutoff': phi_cutoff,
        }
    except Exception as e:
        return {'error': str(e)}
