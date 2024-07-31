from os import path

# Paths
DATA_DIR = path.join(path.dirname(path.abspath(__file__)), "..", "data/")
ESSAYS_DIR = DATA_DIR + "essays/"

# Data Files
ESSAYS = {"mice.csv", "rat.csv", "hepg2.csv"}

CATEGORICAL_COLUMNS = list(
    {
        "ESPÉCIE/LINHAGEM",
        "OUTRO_CRIOPROTETOR",
        "TIPO_DE_MEIO_DE_CULTURA",
        "TEMPO_DE_AVALIAÇÃO_DA_VIABILIDADE",
        "TESTE_DE_VIABILIDADE",
    }
)

NUMERICAL_COLUMNS = list(
    {
        "%_DMSO",
        "%_SFB",
        "%_MEIO_DE_CULTURA",
        "%_OUTRO_CRIOPROTETOR",
        "%_SOLUÇÃO_TOTAL",
        "%_ANTES_DO_CONGELAMENTO",
        "%_APÓS_O_DESCONGELAMENTO",
        "%_QUEDA_DA_VIABILIDADE",
        "TREHALOSE",
        "GLICEROL",
        "SACAROSE",
        "GLICOSE",
        "MALTOSE",
        "LACTOSE",
        "RAFFINOSE",
        "MALTOTRIOSE",
        "MALTOTETRAOSE",
        "MALTOPENTAOSE",
        "MALTOEXAOSE",
        "MALTOHEPTAOSE",
        "ϒ-CYCLODEXTRIN",
        "DEXTRAN",
        "Di-rhamnolipids",
    }
)

COLUMNS_TO_REMOVE = list(
    {  # Removed from model
        "REFERÊNCIA",
        "TEMPO_DE_AVALIAÇÃO_DA_VIABILIDADE",
        "TESTE_DE_VIABILIDADE",
        "ESPÉCIE/LINHAGEM",
    }
)

LABEL_COLUMNS = list(
    {
        "%_ANTES_DO_CONGELAMENTO",
        "%_APÓS_O_DESCONGELAMENTO",
        "%_QUEDA_DA_VIABILIDADE",
    }
)

LABEL = "%_QUEDA_DA_VIABILIDADE"

MODEL_PATH = DATA_DIR + "model.h5"
