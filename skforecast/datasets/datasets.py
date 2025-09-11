
################################################################################
#                            skforecast.datasets                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pandas as pd
import textwrap
from urllib.parse import urlparse
from rich.console import Console
from rich.panel import Panel

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
            'Australian health system had between 1991 and 2008. '
        ),
        'source': (
            'Hyndman R (2023). fpp3: Data for Forecasting: Principles and Practice'
            '(3rd Edition). http://pkg.robjhyndman.com/fpp3package/,'
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
        'freq': 'H',
        'file_type': 'csv',
        'description': (
            'Hourly measures of several air chemical pollutant at Valencia city '
            '(Avd. Francia) from 2019-01-01 to 20213-12-31. Including the following '
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
        'freq': 'H',
        'file_type': 'csv',
        'description': (
            'Hourly measures of several air chemical pollutant at Valencia city '
            '(Avd. Francia) from 2019-01-01 to 20213-12-31. Including the following '
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
        'date_format': '%Y-%m-%d',
        'freq': '1D',
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
        'freq': 'H',
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
        'freq': 'H',
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
        'freq': 'Q',
        'file_type': 'csv',
        'description': (
            "Quarterly overnight trips (in thousands) from 1998 Q1 to 2016 Q4 "
            "across Australia. The tourism regions are formed through the "
            "aggregation of Statistical Local Areas (SLAs) which are defined by "
            "the various State and Territory tourism authorities according to "
            "their research and marketing needs."
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
            'https://github.com/facebook/prophet/blob/{version}/examples/'
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
        'description': 'Half-hourly electricity demand for Victoria, Australia',
        'source': (
            "O'Hara-Wild M, Hyndman R, Wang E, Godahewa R (2022).tsibbledata: Diverse "
            "Datasets for 'tsibble'. https://tsibbledata.tidyverts.org/, "
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
            '2017-12-31 for 50 products (SKU) in 10 stores.'
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
        'description': "Time series with daily frequency from the M4 competition.",
        'source': (
            "Monash Time Series Forecasting Repository  "
            "(https://zenodo.org/communities/forecasting) Godahewa, R., Bergmeir, "
            "C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). Monash "
            "Time Series Forecasting Archive. In Neural Information Processing "
            "Systems Track on Datasets and Benchmarks. \n"
            "Raw data, available in .tsf format, has been converted to Pandas "
            "format using the code provided by the authors in "
            "https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py \n"
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
        'freq': 'H',
        'file_type': 'parquet',
        'description': "Time series with hourly frequency from the M4 competition.",
        'source': (
            "Monash Time Series Forecasting Repository  "
            "(https://zenodo.org/communities/forecasting) Godahewa, R., Bergmeir, "
            "C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). Monash "
            "Time Series Forecasting Archive. In Neural Information Processing "
            "Systems Track on Datasets and Benchmarks. \n"
            "Raw data, available in .tsf format, has been converted to Pandas "
            "format using the code provided by the authors in "
            "https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py \n"
            "The category of each time series has been included in the dataset. This "
            "information has been obtained from the Kaggle competition page: "
            "https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset"
        )
    },
    'ashrae_daily': {
        'url': 'https://drive.google.com/file/d/1fMsYjfhrFLmeFjKG3jenXjDa5s984ThC/view?usp=sharing',
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'parquet',
        'description': (
            "Daily energy consumption data from the ASHRAE competition with "
            "building metadata and weather data."
        ),
        'source': (
            "Kaggle competition Addison Howard, Chris Balbach, Clayton Miller, "
            "Jeff Haberl, Krishnan Gowri, Sohier Dane. (2019). ASHRAE - Great Energy "
            "Predictor III. Kaggle. https://www.kaggle.com/c/ashrae-energy-prediction/overview"
        )
    },
    'bdg2_daily': {
        'url': 'https://drive.google.com/file/d/1KHYopzclKvS1F6Gt6GoJWKnxiuZ2aqen/view?usp=sharing',
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'parquet',
        'description': (
            "Daily energy consumption data from the The Building Data Genome Project 2 "
            "with building metadata and weather data. "
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
        'url': 'https://raw.githubusercontent.com/skforecast/skforecast-datasets/refs/heads/{version}/data/bdg2_daily_sample.csv',
        'sep': ',',
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'csv',
        'description': (
            "Daily energy consumption data of two buildings sampled from the "
            "The Building Data Genome Project 2. "
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
        'url': 'https://drive.google.com/file/d/1I2i5mZJ82Cl_SHPTaWJmLoaXnntdCgh7/view?usp=sharing',
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'H',
        'file_type': 'parquet',
        'description': (
            "Hourly energy consumption data from the The Building Data Genome Project 2 "
            "with building metadata and weather data. "
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
        'url': 'https://raw.githubusercontent.com/skforecast/skforecast-datasets/refs/heads/{version}/data/bdg2_hourly_sample.csv',
        'sep': ',',
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': 'H',
        'file_type': 'csv',
        'description': (
            "Daily energy consumption data of two buildings sampled from the "
            "The Building Data Genome Project 2. "
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
        'url': [
            'https://drive.google.com/file/d/1JOqBsSHegly6iSJFgmkugAko734c6ZW5/view?usp=sharing',
            'https://drive.google.com/file/d/1BhO1BUvs-d7ipXrm7caC3Wd_d0C_6PZ8/view?usp=sharing',
            'https://drive.google.com/file/d/1oHwkQ_QycJVTZMb6bH8C2klQB971gXXA/view?usp=sharing',
            'https://drive.google.com/file/d/1OvYzFlDG04YgTvju2k02vHEOj0nIuwei/view?usp=sharing'
        ],
        'sep': None,
        'index_col': 'timestamp',
        'date_format': '%Y-%m-%d',
        'freq': 'D',
        'file_type': 'parquet',
        'description': (
            "Daily sales data from the M5 competition with product metadata and calendar data."
        ),
        'source': (
            "Addison Howard, inversion, Spyros Makridakis, and vangelis. "
            "M5 Forecasting - Accuracy. https://kaggle.com/competitions/m5-forecasting-accuracy, 2020. Kaggle."
        )
    },
    'ett_m1': {
        'url': (
            'https://raw.githubusercontent.com/skforecast/'
            'skforecast-datasets/refs/heads/{version}/data/ETTm1.csv'
        ),
        'sep': ',',
        'index_col': 'date',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'freq': '15min',
        'file_type': 'csv',
        'description': (
            'Data from an electricity transformer station was collected between July '
            '2016 and July 2018 (2 years x 365 days x 24 hours x 4 intervals per '
            'hour = 70,080 data points). Each data point consists of 8 features, '
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
            'Data from an electricity transformer station was collected between July '
            '2016 and July 2018 (2 years x 365 days x 24 hours x 4 intervals per '
            'hour = 70,080 data points). Each data point consists of 8 features, '
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
            'Data from an electricity transformer station was collected between July '
            '2016 and July 2018 (2 years x 365 days x 24 hours x 4 intervals per '
            'hour = 70,080 data points). Each data point consists of 8 features, '
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
            'https://raw.githubusercontent.com/skforecast/skforecast-datasets/refs/heads/'
            '{version}/data/expenditures_australia.csv'
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
            'https://raw.githubusercontent.com/skforecast/skforecast-datasets/refs/heads/'
            '{version}/data/public-transport-madrid.csv'
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
            'https://raw.githubusercontent.com/skforecast/skforecast-datasets/refs/heads/'
            '{version}/data/turbine_emission.csv'
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


def show_datasets_info(
    datasets_names: list[str] | None = None,
    version: str = 'latest'
) -> None:
    """
    Print information about available datasets. If `datasets_names` is provided,
    only information about those datasets will be printed.

    Parameters
    ----------
    datasets_names: list, default None
        List of dataset names to display information about. If None, information 
        about all datasets will be displayed.
    version: str
        Version of the datasets to display information about.

    Returns
    -------
    None

    """

    datasets_names = datasets_names or sorted(datasets.keys())
    version = 'main' if version == 'latest' else f'{version}'

    for dataset_name in datasets_names:
        if dataset_name in datasets:
            _print_dataset_info(dataset_name, version=version)
        else:
            print(
                f"Dataset '{dataset_name}' not available. Set argument "
                f"`datasets_names` to None to see all available datasets."
            )


def _print_dataset_info(
    dataset_name: str,
    version: str | int = 'latest',
    shape: tuple[int, int] | None = None
) -> None:
    """
    Display information about the dataset in a card-style panel.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to display information about.
    version: str, int, default `'latest'`
        Version of the dataset. If 'latest', the latest version will be used
        (the one in the main branch). For a list of available versions, see
        the repository branches.
    shape: tuple, default None
        Shape of the dataset (rows, columns). If provided, it will be
        displayed in the panel.

    Returns
    -------
    None

    """

    version = 'main' if version == 'latest' else f'{version}'
    info = datasets.get(dataset_name, None)
    if info is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets are: {list(datasets.keys())}"
        )
    
    console = Console()
    description = textwrap.fill(info['description'], width=80)
    source = info['source']
    source = textwrap.fill(source, width=80)
    url = info['url']
    if isinstance(url, list):
        url = "Data is stored in multiple files:\n  " + "\n  ".join(url)
    if '{version}' in url:
        url = url.format(version=version)
    url = textwrap.fill(url, width=80)
    content = (
        f"[bold]Description:[/bold]\n{description}\n\n"
        f"[bold]Source:[/bold]\n{source}\n\n"
        f"[bold]URL:[/bold]\n{url}"
    )

    if shape is not None:
        content += f"\n\n[bold]Shape:[/bold] {shape[0]} rows x {shape[1]} columns"

    console.print(Panel(content, title=f"[bold]{dataset_name}[/bold]", expand=False))


def fetch_dataset(
    name: str,
    version: str | int = 'latest',
    raw: bool = False,
    kwargs_read_csv: dict = {},
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch a dataset from the skforecast-datasets repository.

    Parameters
    ----------
    name: str
        Name of the dataset to fetch.
    version: str, int, default `'latest'`
        Version of the dataset to fetch. If 'latest', the latest version will be 
        fetched (the one in the main branch). For a list of available versions, 
        see the repository branches.
    raw: bool, default `False`
        If True, the raw dataset is fetched. If False, the preprocessed dataset 
        is fetched. The preprocessing consists of setting the column with the 
        date/time as index and converting the index to datetime. A frequency is 
        also set to the index.
    kwargs_read_csv: dict, default `{}`
        Kwargs to pass to pandas `read_csv` function.
    verbose: bool, default `True`
        If True, print information about the dataset.
    
    Returns
    -------
    df: pandas DataFrame
        Dataset.
    
    """

    version = 'main' if version == 'latest' else f'{version}'
    
    if name not in datasets.keys():
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets are: {sorted(datasets.keys())}"
        )
    
    url = datasets[name]['url']
    if '{version}' in url:
        url = url.format(version=version)
    file_type = datasets[name]['file_type']

    if not isinstance(url, list):
        parsed = urlparse(url)
        if parsed.scheme == "https" and parsed.netloc == "drive.google.com":
            file_id = url.split('/')[-2]
            url = 'https://drive.google.com/uc?id=' + file_id
        if file_type == 'csv':
            try:
                sep = datasets[name]['sep']
                df = pd.read_csv(url, sep=sep, **kwargs_read_csv)
            except Exception as e:
                raise ValueError(
                    f"Error reading dataset '{name}' from {url}: {str(e)}."
                )
        if file_type == 'parquet':
            try:
                df = pd.read_parquet(url)
            except Exception as e:
                raise ValueError(
                    f"Error reading dataset '{name}' from {url}: {str(e)}."
                )
    else:
        try: 
            df = []
            for url_partition in url:
                path = 'https://drive.google.com/uc?export=download&id=' + url_partition.split('/')[-2]
                df.append(pd.read_parquet(path))
        except Exception as e:
            raise ValueError(
                f"Error reading dataset '{name}' from {url}: {str(e)}."
            )
        df = pd.concat(df, axis=0).reset_index(drop=True)

    if not raw:
        try:
            index_col = datasets[name]['index_col']
            freq = datasets[name]['freq']
            if freq == 'H' and pd.__version__ >= '2.2.0':
                freq = "h"
            date_format = datasets[name]['date_format']
            df = df.set_index(index_col)
            df.index = pd.to_datetime(df.index, format=date_format)
            df = df.asfreq(freq)
            df = df.sort_index()
        except:
            pass
    
    if verbose:
        _print_dataset_info(name, version=version, shape=df.shape)

    return df


def load_demo_dataset(version: str = 'latest') -> pd.Series:
    """
    Load demo data set with monthly expenditure ($AUD) on corticosteroid drugs that
    the Australian health system had between 1991 and 2008. Obtained from the book:
    Forecasting: Principles and Practice by Rob J Hyndman and George Athanasopoulos.
    Index is set to datetime with monthly frequency and sorted.

    Parameters
    ----------
    version: str, default `'latest'`
        Version of the dataset to fetch. If 'latest', the latest version will be
        fetched (the one in the main branch). For a list of available versions,
        see the repository branches.

    Returns
    -------
    df: pandas Series
        Dataset.
    
    """

    version = 'main' if version == 'latest' else f'{version}'

    url = (
        f'https://raw.githubusercontent.com/skforecast/skforecast-datasets/{version}/'
        'data/h2o.csv'
    )

    df = pd.read_csv(url, sep=',', header=0, names=['y', 'datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df = df.set_index('datetime')
    df = df.asfreq('MS')
    df = df['y']
    df = df.sort_index()

    return df
