
################################################################################
#                            skforecast.datasets                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################


from __future__ import annotations
import warnings
import pandas as pd
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from ..exceptions import MissingValuesWarning

# Parquet entries: date_format is ignored, the column is already datetime64 in
# the file.
datasets = {
    'h2o': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/h2o.csv'
        ),
        'sep': ',',
        'index_col': 'fecha',
        'date_format': '%Y-%m-%d',
        'freq': 'MS',
        'file_type': 'csv',
        'description': (
            'Monthly expenditure ($AUD) on corticosteroid drugs that the '
            'Australian health system had between 1991 and 2008.'
        ),
        'source': (
            'Hyndman R (2023). fpp3: Data for Forecasting: Principles and Practice '
            '(3rd Edition). http://pkg.robjhyndman.com/fpp3package/, '
            'https://github.com/robjhyndman/fpp3package, http://OTexts.com/fpp3.'
        )
    },
    'h2o_exog': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/h2o_exog.csv'
        ),
        'sep': ',',
        'index_col': 'fecha',
        'date_format': '%Y-%m-%d',
        'freq': 'MS',
        'file_type': 'csv',
        'description': (
            'Monthly expenditure ($AUD) on corticosteroid drugs that the '
            'Australian health system had between 1991 and 2008. Two additional '
            'variables (exog_1, exog_2) are simulated.'
        ),
        'source': (
            "Hyndman R (2023). fpp3: Data for Forecasting: Principles and Practice "
            "(3rd Edition). http://pkg.robjhyndman.com/fpp3package/, "
            "https://github.com/robjhyndman/fpp3package, http://OTexts.com/fpp3."
        )
    },
    'fuel_consumption': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/consumos-combustibles-mensual.csv'
        ),
        'sep': ',',
        'index_col': 'Fecha',
        'date_format': '%Y-%m-%d',
        'freq': 'MS',
        'file_type': 'csv',
        'description': (
            'Monthly fuel consumption in Spain from 1969-01-01 to 2022-08-01.'
        ),
        'source': (
            'Obtained from Corporación de Reservas Estratégicas de Productos '
            'Petrolíferos and Corporación de Derecho Público tutelada por el '
            'Ministerio para la Transición Ecológica y el Reto Demográfico. '
            'https://www.cores.es/es/estadisticas'
        )
    },
    'items_sales': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/simulated_items_sales.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'csv',
        'description': 'Simulated time series for the sales of 3 different items.',
        'source': 'Simulated data.'
    },
    'air_quality_valencia': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/air_quality_valencia.csv'
        ),
        'sep': ',',
        'index_col': 'datetime',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'csv',
        'description': (
            'Hourly measures of several air chemical pollutants at Valencia city '
            '(Avd. Francia) from 2019-01-01 to 2023-12-31. Including the following '
            'variables: pm2.5 (µg/m³), CO (mg/m³), NO (µg/m³), NO2 (µg/m³), '
            'PM10 (µg/m³), NOx (µg/m³), O3 (µg/m³), Veloc. (m/s), Direc. (degrees), '
            'SO2 (µg/m³).'
        ),
        'source': (
            "Red de Vigilancia y Control de la Contaminación Atmosférica, "
            "46250047-València - Av. França, "
            "https://mediambient.gva.es/es/web/calidad-ambiental/datos-historicos."
        )
    },
    'air_quality_valencia_no_missing': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/air_quality_valencia_no_missing.csv'
        ),
        'sep': ',',
        'index_col': 'datetime',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'csv',
        'description': (
            'Hourly measures of several air chemical pollutants at Valencia city '
            '(Avd. Francia) from 2019-01-01 to 2023-12-31. Including the following '
            'variables: pm2.5 (µg/m³), CO (mg/m³), NO (µg/m³), NO2 (µg/m³), '
            'PM10 (µg/m³), NOx (µg/m³), O3 (µg/m³), Veloc. (m/s), Direc. (degrees), '
            'SO2 (µg/m³). Missing values have been imputed using linear interpolation.'
        ),
        'source': (
            "Red de Vigilancia y Control de la Contaminación Atmosférica, "
            "46250047-València - Av. França, "
            "https://mediambient.gva.es/es/web/calidad-ambiental/datos-historicos."
        )
    },
    'website_visits': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/visitas_por_dia_web_cienciadedatos.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%d/%m/%y',
        'freq': 'D',
        'file_type': 'csv',
        'description': (
            'Daily visits to the cienciadedatos.net website registered with the '
            'google analytics service.'
        ),
        'source': (
            "Amat Rodrigo, J. (2021). cienciadedatos.net (1.0.0). Zenodo. "
            "https://doi.org/10.5281/zenodo.10006330"
        )
    },
    'bike_sharing': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/bike_sharing_dataset_clean.csv'
        ),
        'sep': ',',
        'index_col': 'date_time',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'csv',
        'description': (
            'Hourly usage of the bike share system in the city of Washington D.C. '
            'during the years 2011 and 2012. In addition to the number of users per '
            'hour, information about weather conditions and holidays is available.'
        ),
        'source': (
            "Fanaee-T,Hadi. (2013). Bike Sharing Dataset. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C5W894."
        )
    },
    'bike_sharing_extended_features': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/bike_sharing_extended_features.csv'
        ),
        'sep': ',',
        'index_col': 'date_time',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'csv',
        'description': (
            'Hourly usage of the bike share system in the city of Washington D.C. '
            'during the years 2011 and 2012. In addition to the number of users per '
            'hour, the dataset was enriched by introducing supplementary features. '
            'Addition includes calendar-based variables (day of the week, hour of '
            'the day, month, etc.), indicators for sunlight, incorporation of '
            'rolling temperature averages, and the creation of polynomial features '
            'generated from variable pairs. All cyclic variables are encoded using '
            'sine and cosine functions to ensure accurate representation.'
        ),
        'source': (
            "Fanaee-T,Hadi. (2013). Bike Sharing Dataset. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C5W894."
        )
    },
    'australia_tourism': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/australia_tourism.csv'
        ),
        'sep': ',',
        'index_col': 'date_time',
        'date_format': '%Y-%m-%d',
        'freq': 'QS',
        'file_type': 'csv',
        'description': (
            "Quarterly overnight trips (in thousands) from 1998 Q1 to 2017 Q4 "
            "across Australia. The tourism regions are formed through the "
            "aggregation of Statistical Local Areas (SLAs) which are defined by "
            "the various State and Territory tourism authorities according to "
            "their research and marketing needs. Long format: one row per "
            "region, state, purpose and quarter (304 series)."
        ),
        'source': (
            "Wang, E, D Cook, and RJ Hyndman (2020). A new tidy data structure to "
            "support exploration and modeling of temporal data, Journal of "
            "Computational and Graphical Statistics, 29:3, 466-478, "
            "doi:10.1080/10618600.2019.1695624."
        )
    },
    'uk_daily_flights': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/uk_daily_flights.csv'
        ),
        'sep': ',',
        'index_col': 'Date',
        'date_format': '%d/%m/%Y',
        'freq': 'D',
        'file_type': 'csv',
        'description': 'Daily number of flights in UK from 02/01/2019 to 23/01/2022.',
        'source': (
            'Experimental statistics published as part of the Economic activity and '
            'social change in the UK, real-time indicators release, Published 27 '
            'January 2022. Daily flight numbers are available in the dashboard '
            'provided by the European Organisation for the Safety of Air Navigation '
            '(EUROCONTROL). '
            'https://www.ons.gov.uk/economy/economicoutputandproductivity/output/'
            'bulletins/economicactivityandsocialchangeintheukrealtimeindicators/latest'
        )
    },
    'wikipedia_visits': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/wikipedia_visits.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'csv',
        'description': (
            'Log daily page views for the Wikipedia page for Peyton Manning. '
            'Scraped data using the Wikipediatrend package in R.'
        ),
        'source': (
            'https://github.com/facebook/prophet/blob/main/examples/'
            'example_wp_log_peyton_manning.csv'
        )
    },
    'vic_electricity': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/vic_electricity.csv'
        ),
        'sep': ',',
        'index_col': 'Time',
        'date_format': '%Y-%m-%dT%H:%M:%SZ',
        'freq': '30min',
        'file_type': 'csv',
        'description': (
            'Half-hourly electricity demand for Victoria, Australia. `Time` is in '
            'UTC (suffix Z, parsed as a naive datetime); `Date` and `Holiday` refer '
            'to local time in Melbourne (UTC+10, UTC+11 in summer).'
        ),
        'source': (
            "O'Hara-Wild M, Hyndman R, Wang E, Godahewa R (2022). tsibbledata: "
            "Diverse Datasets for 'tsibble'. https://tsibbledata.tidyverts.org/, "
            "https://github.com/tidyverts/tsibbledata/. "
            "https://tsibbledata.tidyverts.org/reference/vic_elec.html"
        )
    },
    'vic_electricity_classification': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/vic_electricity_classification.csv'
        ),
        'sep': ',',
        'index_col': 'Time',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'csv',
        'description': (
            "Hourly electricity demand for Victoria, Australia classified into three categories: "
            "'low', 'medium' and 'high' according to the 20th and 80th percentiles. "
            "The dataset also includes temperature, holiday indicator and hour of the day as features."
        ),
        'source': (
            "O'Hara-Wild M, Hyndman R, Wang E, Godahewa R (2022). tsibbledata: "
            "Diverse Datasets for 'tsibble'. https://tsibbledata.tidyverts.org/, "
            "https://github.com/tidyverts/tsibbledata/. "
            "https://tsibbledata.tidyverts.org/reference/vic_elec.html"
        )
    },
    'store_sales': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/store_sales.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'csv',
        'description': (
            'This dataset contains 913,000 sales transactions from 2013-01-01 to '
            '2017-12-31 for 50 products (SKU) in 10 stores. Long format: one row '
            'per store, item and date.'
        ),
        'source': (
            'The original data was obtained from: inversion. (2018). Store Item '
            'Demand Forecasting Challenge. Kaggle. '
            'https://kaggle.com/competitions/demand-forecasting-kernels-only'
        )
    },
    'bicimad': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/bicimad_users.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'csv',
        'description': (
            'This dataset contains the daily users of the bicycle rental '
            'service (BiciMad) in the city of Madrid (Spain) from 2014-06-23 '
            'to 2022-09-30.'
        ),
        'source': (
            'The original data was obtained from: Portal de datos abiertos '
            'del Ayuntamiento de Madrid https://datos.madrid.es/portal/site/egob'
        )
    },
    'm4_daily': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/m4_daily.parquet'
        ),
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'D',
        'file_type': 'parquet',
        'description': (
            "Time series with daily frequency from the M4 competition. Timestamps "
            "are artificial (the M4 series have no real dates): all at 12:00 and "
            "spanning 1959 to 2039. Long format: one row per series and timestamp."
        ),
        'source': (
            "Monash Time Series Forecasting Repository "
            "(https://zenodo.org/communities/forecasting) Godahewa, R., Bergmeir, "
            "C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). Monash "
            "Time Series Forecasting Archive. In Neural Information Processing "
            "Systems Track on Datasets and Benchmarks.\n"
            "Raw data, available in .tsf format, has been converted to Pandas "
            "format using the code provided by the authors in "
            "https://github.com/rakshitha123/TSForecasting/blob/master/utils/"
            "data_loader.py\n"
            "The category of each time series has been included in the dataset. This "
            "information has been obtained from the Kaggle competition page: "
            "https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset"
        )
    },
    'm4_hourly': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/m4_hourly.parquet'
        ),
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'parquet',
        'description': (
            "Time series with hourly frequency from the M4 competition. Timestamps "
            "are artificial (the M4 series have no real dates): every series starts "
            "at 12:00, and they span 2009 to 2017. Long format: one row per series "
            "and timestamp."
        ),
        'source': (
            "Monash Time Series Forecasting Repository "
            "(https://zenodo.org/communities/forecasting) Godahewa, R., Bergmeir, "
            "C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). Monash "
            "Time Series Forecasting Archive. In Neural Information Processing "
            "Systems Track on Datasets and Benchmarks.\n"
            "Raw data, available in .tsf format, has been converted to Pandas "
            "format using the code provided by the authors in "
            "https://github.com/rakshitha123/TSForecasting/blob/master/utils/"
            "data_loader.py\n"
            "The category of each time series has been included in the dataset. This "
            "information has been obtained from the Kaggle competition page: "
            "https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset"
        )
    },
    'ashrae_daily': {
        'url': 'https://huggingface.co/datasets/skforecast/ashrae_daily/resolve/main/ashrae_daily.parquet',
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'parquet',
        'description': (
            "Daily energy consumption data from the ASHRAE competition with "
            "building metadata and weather data. Long format: one row per "
            "building and date."
        ),
        'source': (
            "Kaggle competition Addison Howard, Chris Balbach, Clayton Miller, "
            "Jeff Haberl, Krishnan Gowri, Sohier Dane. (2019). ASHRAE - Great Energy "
            "Predictor III. Kaggle. https://www.kaggle.com/c/ashrae-energy-prediction/overview"
        )
    },
    'bdg2_daily': {
        'url': 'https://huggingface.co/datasets/skforecast/bdg2_daily/resolve/main/bdg2_daily.parquet',
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'parquet',
        'description': (
            "Daily energy consumption data from the Building Data Genome Project 2 "
            "with building metadata and weather data. Long format: one row per "
            "building and date. "
            "https://github.com/buds-lab/building-data-genome-project-2"
        ),
        'source': (
            "Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data "
            "Genome Project 2, energy meter data from the ASHRAE Great Energy "
            "Predictor III competition. Sci Data 7, 368 (2020). "
            "https://doi.org/10.1038/s41597-020-00712-x"
        )
    },
    'bdg2_daily_sample': {
        'url': 'https://raw.githubusercontent.com/skforecast/skforecast-datasets/{version}/data/bdg2_daily_sample.csv',
        'sep': ',',
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'csv',
        'description': (
            "Daily energy consumption data of two buildings sampled from the "
            "Building Data Genome Project 2. "
            "https://github.com/buds-lab/building-data-genome-project-2"
        ),
        'source': (
            "Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data "
            "Genome Project 2, energy meter data from the ASHRAE Great Energy "
            "Predictor III competition. Sci Data 7, 368 (2020). "
            "https://doi.org/10.1038/s41597-020-00712-x"
        )
    },
    'bdg2_hourly': {
        'url': 'https://huggingface.co/datasets/skforecast/bdg2_hourly/resolve/main/bdg2_hourly.parquet',
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'parquet',
        'description': (
            "Hourly energy consumption data from the Building Data Genome Project 2 "
            "with building metadata and weather data. Long format: one row per "
            "building and timestamp. "
            "https://github.com/buds-lab/building-data-genome-project-2"
        ),
        'source': (
            "Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data "
            "Genome Project 2, energy meter data from the ASHRAE Great Energy "
            "Predictor III competition. Sci Data 7, 368 (2020). "
            "https://doi.org/10.1038/s41597-020-00712-x"
        )
    },
    'bdg2_hourly_sample': {
        'url': 'https://raw.githubusercontent.com/skforecast/skforecast-datasets/{version}/data/bdg2_hourly_sample.csv',
        'sep': ',',
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'csv',
        'description': (
            "Hourly energy consumption data of two buildings sampled from the "
            "Building Data Genome Project 2. "
            "https://github.com/buds-lab/building-data-genome-project-2"
        ),
        'source': (
            "Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data "
            "Genome Project 2, energy meter data from the ASHRAE Great Energy "
            "Predictor III competition. Sci Data 7, 368 (2020). "
            "https://doi.org/10.1038/s41597-020-00712-x"
        )
    },
    'm5': {
        'url': 'https://huggingface.co/datasets/skforecast/m5_daily/resolve/main/m5_daily.parquet',
        'sep': None,
        'index_col': 'datetime',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'parquet',
        'description': (
            "Daily sales data from the M5 competition with product metadata and "
            "calendar data. Series ids keep the `_evaluation` suffix of the Kaggle "
            "files; `sell_price` is NaN before the product was first sold. Long "
            "format: one row per series and date."
        ),
        'source': (
            "Addison Howard, inversion, Spyros Makridakis, and vangelis. "
            "M5 Forecasting - Accuracy. https://kaggle.com/competitions/m5-forecasting-accuracy, 2020. Kaggle."
        )
    },
    'ett_m1': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/ETTm1.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': '15min',
        'file_type': 'csv',
        'description': (
            'Data from an electricity transformer station was collected from '
            '2016-07-01 to 2018-06-26 (69,680 data points every 15 minutes). '
            'Each data point consists of 8 features, '
            'including the date of the point, the predictive value "Oil Temperature (OT)", '
            'and 6 different types of external power load features: High UseFul Load (HUFL), '
            'High UseLess Load (HULL), Middle UseFul Load (MUFL), Middle UseLess Load (MULL), '
            'Low UseFul Load (LUFL), Low UseLess Load (LULL).'
        ),
        'source': (
            'Zhou, Haoyi & Zhang, Shanghang & Peng, Jieqi & Zhang, Shuai & Li, '
            'Jianxin & Xiong, Hui & Zhang, Wancai. (2020). Informer: Beyond Efficient '
            'Transformer for Long Sequence Time-Series Forecasting. '
            '[10.48550/arXiv.2012.07436](https://arxiv.org/abs/2012.07436). '
            'https://github.com/zhouhaoyi/ETDataset'
        )
    },
    'ett_m2': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/ETTm2.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': '15min',
        'file_type': 'csv',
        'description': (
            'Data from an electricity transformer station was collected from '
            '2016-07-01 to 2018-06-26 (69,680 data points every 15 minutes). '
            'Each data point consists of 8 features, '
            'including the date of the point, the predictive value "Oil Temperature (OT)", '
            'and 6 different types of external power load features: High UseFul Load (HUFL), '
            'High UseLess Load (HULL), Middle UseFul Load (MUFL), Middle UseLess Load (MULL), '
            'Low UseFul Load (LUFL), Low UseLess Load (LULL).'
        ),
        'source': (
            'Zhou, Haoyi & Zhang, Shanghang & Peng, Jieqi & Zhang, Shuai & Li, '
            'Jianxin & Xiong, Hui & Zhang, Wancai. (2020). Informer: Beyond Efficient '
            'Transformer for Long Sequence Time-Series Forecasting. '
            '[10.48550/arXiv.2012.07436](https://arxiv.org/abs/2012.07436). '
            'https://github.com/zhouhaoyi/ETDataset'
        )
    },
    'ett_m2_extended': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/ETTm2_extended.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': '15min',
        'file_type': 'csv',
        'description': (
            'Data from an electricity transformer station was collected from '
            '2016-07-01 to 2018-06-26 (69,680 data points every 15 minutes). '
            'Each data point consists of 8 features, '
            'including the date of the point, the predictive value "Oil Temperature (OT)", '
            'and 6 different types of external power load features: High UseFul Load (HUFL), '
            'High UseLess Load (HULL), Middle UseFul Load (MUFL), Middle UseLess Load (MULL), '
            'Low UseFul Load (LUFL), Low UseLess Load (LULL). Additional variables are '
            'created based on calendar information (year, month, week, day of the week, '
            'and hour). These variables have been encoded using the cyclical encoding '
            'technique (sin and cos transformations) to preserve the cyclical nature '
            'of the data.'
        ),
        'source': (
            'Zhou, Haoyi & Zhang, Shanghang & Peng, Jieqi & Zhang, Shuai & Li, '
            'Jianxin & Xiong, Hui & Zhang, Wancai. (2020). Informer: Beyond Efficient '
            'Transformer for Long Sequence Time-Series Forecasting. '
            '[10.48550/arXiv.2012.07436](https://arxiv.org/abs/2012.07436). '
            'https://github.com/zhouhaoyi/ETDataset'
        )
    },
    'expenditures_australia': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/expenditures_australia.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d',
        'freq': 'MS',
        'file_type': 'csv',
        'description': (
            'Monthly expenditure on cafes, restaurants and takeaway food services '
            'in Victoria (Australia) from April 1982 up to April 2024.'
        ),
        'source': (
            'Australian Bureau of Statistics. Catalogue No. 8501.0 '
            'https://www.abs.gov.au/statistics/industry/retail-and-wholesale-trade/'
            'retail-trade-australia/apr-2024/8501011.xlsx'
        )
    },
    'public_transport_madrid': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/public-transport-madrid.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'csv',
        'description': (
            'Daily users of public transport in Madrid (Spain) from 2023-01-01 to 2024-12-15.'
        ),
        'source': (
            'Consorcio Regional de Transportes de Madrid CRTM, CRTM Evolucion demanda diaria '
            'https://datos.crtm.es/documents/a7210254c4514a19a51b1617cfd61f75/about'
        )
    },
    'turbine_emission': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/{version}/data/turbine_emission.csv'
        ),
        'sep': ',',
        'index_col': 'datetime',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'h',
        'file_type': 'csv',
        'description': (
            'The dataset contains 36733 instances of 11 sensor measures aggregated over one hour, '
            'from a gas turbine located in Turkey for the purpose of studying flue gas emissions, '
            'namely CO and NOx. Available variables include: Ambient temperature (AT), '
            'Ambient pressure (AP), Ambient humidity (AH), Air filter difference pressure (AFDP), '
            'Gas turbine exhaust pressure (GTEP), Turbine inlet temperature (TIT), '
            'Turbine after temperature (TAT), Compressor discharge pressure (CDP), Turbine energy yield (TEY), '
            'Carbon monoxide (CO), and Nitrogen oxides (NOx). Covered period from 2011-01-01 00:00:00 '
            'to 2015-03-11 12:00:00.'
        ),
        'source': (
            'https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set'
        )
    },
}


def _warn_deprecated_version(version: str | int | None) -> None:
    """
    Warn that `version` is deprecated when it is passed.

    Parameters
    ----------
    version : str, int, None
        Value of the deprecated `version` argument of the public function.

    Returns
    -------
    None

    """

    # TODO: Remove in version 0.29.0 together with the `version` argument. Then
    # make keyword-only the arguments that follow it (`fetch_dataset(name, *,
    # raw=False, ...)`, `load_demo_dataset(*, verbose=True)`), otherwise a
    # positional `fetch_dataset('h2o', 'latest')` would pass 'latest' to `raw`.
    if version is not None:
        warnings.warn(
            "`version` is deprecated since version 0.27.0 and has no effect: "
            "datasets are always fetched from the main branch of "
            "skforecast-datasets. It will be removed in version 0.29.0.",
            FutureWarning,
            stacklevel=3
        )


def show_datasets_info(
    datasets_names: str | list[str] | None = None,
    version: str | int | None = None
) -> None:
    """
    Print information about available datasets. If `datasets_names` is provided,
    only information about those datasets will be printed.

    Parameters
    ----------
    datasets_names : str, list, default None
        Name or list of names of the datasets to display information about. If
        None, information about all datasets will be displayed.
    version : str, int, default None
        **Deprecated** since version 0.27.0, it has no effect and will be removed
        in version 0.29.0. Datasets are always fetched from the main branch of
        skforecast-datasets.

    Returns
    -------
    None

    """

    _warn_deprecated_version(version)
    if datasets_names is None:
        datasets_names = sorted(datasets)
    elif isinstance(datasets_names, str):
        datasets_names = [datasets_names]
    else:
        datasets_names = list(datasets_names)

    for dataset_name in datasets_names:
        if dataset_name in datasets:
            _print_dataset_info(dataset_name)
        else:
            print(
                f"Dataset '{dataset_name}' not available. Set argument "
                f"`datasets_names` to None to see all available datasets."
            )


def _print_dataset_info(
    dataset_name: str,
    shape: tuple[int, int] | None = None
) -> None:
    """
    Display information about the dataset in a card-style panel.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to display information about.
    shape : tuple, default None
        Shape of the dataset (rows, columns). If provided, it will be
        displayed in the panel.

    Returns
    -------
    None

    """

    info = datasets.get(dataset_name, None)
    if info is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets are: {sorted(datasets)}"
        )
    
    console = Console()
    url = info['url']
    if '{version}' in url:
        url = url.format(version='main')
    # rich wraps the text to the width of the console
    content = (
        f"[bold]Description:[/bold]\n{escape(info['description'])}\n\n"
        f"[bold]Source:[/bold]\n{escape(info['source'])}\n\n"
        f"[bold]URL:[/bold]\n{escape(url)}"
    )

    if shape is not None:
        if len(shape) == 1:
            content += f"\n\n[bold]Shape:[/bold] {shape[0]} rows"
        else:
            content += f"\n\n[bold]Shape:[/bold] {shape[0]} rows x {shape[1]} columns"

    console.print(Panel(content, title=f"[bold]{dataset_name}[/bold]", expand=False))


def fetch_dataset(
    name: str,
    version: str | int | None = None,
    raw: bool = False,
    kwargs_read: dict | None = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch a dataset from the skforecast-datasets repository.

    Parameters
    ----------
    name : str
        Name of the dataset to fetch.
    version : str, int, default None
        **Deprecated** since version 0.27.0, it has no effect and will be removed
        in version 0.29.0. Datasets are always fetched from the main branch of
        skforecast-datasets.
    raw : bool, default False
        If `True`, the raw dataset is fetched. If `False`, the preprocessed
        dataset is fetched: the column with the date/time is set as index,
        converted to datetime and sorted. When the dates are unique, the
        frequency of the dataset is set with `asfreq` (rows with NaN are
        introduced for missing dates, with a `MissingValuesWarning`).
    kwargs_read : dict, default None
        Kwargs to pass to pandas `pd.read_csv` or `pd.read_parquet` function
        depending on the dataset file type.
    verbose : bool, default True
        If `True`, print information about the dataset.

    Returns
    -------
    df : pandas DataFrame
        Dataset.

    Notes
    -----
    Datasets with several series in long format (one row per series and date,
    see their description in `show_datasets_info`) have repeated dates in the
    index, so no frequency is set and the rows are sorted by date, keeping the
    order of the series within each date. Use `reshape_series_long_to_dict`
    (after `reset_index`, it needs the dates as a column) or `pivot_table` to
    get one series per key or column.

    """

    _warn_deprecated_version(version)
    kwargs_read = kwargs_read or {}

    if name not in datasets:
        raise ValueError(
            f"Dataset '{name}' not found. "
            f"Available datasets are: {sorted(datasets)}"
        )

    file_type = datasets[name]['file_type']
    if file_type not in ('csv', 'parquet'):
        raise ValueError(
            f"Unknown file type '{file_type}' for dataset '{name}'. "
            f"Supported file types are 'csv' and 'parquet'."
        )

    url = datasets[name]['url']
    if '{version}' in url:
        url = url.format(version='main')

    if file_type == 'csv':
        kwargs_read = {'sep': datasets[name]['sep'], **kwargs_read}
        try:
            df = pd.read_csv(url, **kwargs_read)
        except Exception as e:
            raise ValueError(
                f"Error reading dataset '{name}' from {url}: {str(e)}."
            ) from e
    elif file_type == 'parquet':
        try:
            df = pd.read_parquet(url, **kwargs_read)
        except ImportError as e:
            raise ImportError(
                f"Reading dataset '{name}' requires pyarrow or fastparquet: "
                f"pip install pyarrow. Original error: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error reading dataset '{name}' from {url}: {str(e)}."
            ) from e
        # pyarrow keeps the memory used while reading in its pool. Returning it
        # to the OS lets the preprocessing of large datasets (e.g. m5) fit in
        # memory.
        try:
            import pyarrow as pa
            pa.default_memory_pool().release_unused()
        except ImportError:
            pass

    if not raw:
        n_inserted = n_dropped = 0
        try:
            index_col = datasets[name]['index_col']
            freq = datasets[name]['freq']
            date_format = datasets[name]['date_format']
            if df.index.name != index_col:
                # Unlike `set_index`, `pop` does not copy the whole DataFrame
                df.index = df.pop(index_col)
            df.index = pd.to_datetime(df.index, format=date_format)
            if df.index.is_unique:
                index_before = df.index
                df = df.asfreq(freq)
                n_inserted = len(df.index.difference(index_before))
                n_dropped = len(index_before.difference(df.index))
            if not df.index.is_monotonic_increasing:
                # Stable, to keep the order of the series of long-format datasets
                # within each date
                df = df.sort_index(kind='stable')
        except Exception as e:
            warnings.warn(
                f"Could not preprocess dataset '{name}': {e}",
                stacklevel=2
            )

        # Outside the `try`, so that a warning turned into an error is not
        # reported as a preprocessing failure
        if n_inserted > 0:
            warnings.warn(
                f"Dataset '{name}' has {n_inserted} missing dates for frequency "
                f"'{freq}'. Rows with NaN values have been introduced after "
                f"setting the frequency.",
                MissingValuesWarning,
                stacklevel=2
            )
        if n_dropped > 0:
            warnings.warn(
                f"{n_dropped} rows of dataset '{name}' are not aligned with "
                f"frequency '{freq}' and have been dropped after setting the "
                f"frequency. The frequency of the dataset may be wrong, use "
                f"`raw=True` to get the original data.",
                stacklevel=2
            )

    if verbose:
        _print_dataset_info(name, shape=df.shape)

    return df


def load_demo_dataset(
    version: str | int | None = None,
    verbose: bool = True
) -> pd.Series:
    """
    Load demo data set with monthly expenditure ($AUD) on corticosteroid drugs that
    the Australian health system had between 1991 and 2008. Obtained from the book:
    Forecasting: Principles and Practice by Rob J Hyndman and George Athanasopoulos.
    Index is set to datetime with monthly frequency and sorted.

    Parameters
    ----------
    version : str, int, default None
        **Deprecated** since version 0.27.0, it has no effect and will be removed
        in version 0.29.0. Datasets are always fetched from the main branch of
        skforecast-datasets.
    verbose : bool, default True
        If `True`, print information about the dataset.

    Returns
    -------
    y : pandas Series
        Dataset.

    """

    _warn_deprecated_version(version)

    # `version` is not passed, so the deprecation warning is raised only once
    y = fetch_dataset('h2o', raw=False, verbose=False)['x']
    y = y.rename('y')
    y.index.name = 'datetime'

    if verbose:
        _print_dataset_info('h2o', shape=y.shape)

    return y
