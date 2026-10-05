## Devlog Docker Preditor Ethereum

1. Desenhar a arquitetura: faça um esboço UML que represente os componentes da solução e a troca de dados entre eles. Inclua, no mínimo, o ambiente de treinamento, o artefato gerado pelo modelo, o container de inferência/backend e a aplicação cliente. Indique como o modelo treinado chega ao container de inferência.
2. Treinar e exportar o modelo: execute o treinamento em um container Docker ou notebook e salve o modelo em um formato que possa ser carregado posteriormente.
3. Preparar a inferência: implemente em Python o backend do container de inferência. Ele deve carregar o artefato treinado e disponibilizar uma operação para solicitar uma predição. Inclua também uma forma simples de verificar se o serviço está ativo.
4. Integrar e testar: execute os componentes necessários e demonstre que uma solicitação chega ao backend e retorna uma predição. Registre os comandos usados e os resultados observados.
5. Manter o devlog: documente, ao longo do desenvolvimento, as decisões, etapas realizadas, dificuldades, alterações e testes. Inclua o diagrama UML e evidências suficientes para outra pessoa compreender e reproduzir a solução.

Primeiro pedi pra o meu Sonnet 5.5 para desenhar o diagrama conforme minha arquitetura

```mermaid
flowchart LR
    subgraph KAGGLE["Fonte de dados"]
        DS[("Kaggle<br/>varpit94/ethereum-data<br/>ETH-USD.csv")]
    end

    subgraph TRAIN["Container: treinamento (Docker / notebook)"]
        PRE["Pré-processamento<br/>pandas: limpeza, features<br/>(lags, médias móveis)"]
        FIT["Treino<br/>scikit-learn<br/>(ex.: RandomForest / LinearRegression)"]
        EVAL["Avaliação<br/>MAE / RMSE / R²"]
        PRE --> FIT --> EVAL
    end

    subgraph ART["Artefatos (volume Docker compartilhado)"]
        MODEL[["model.joblib"]]
        META[["metadata.json<br/>features, versão, métricas"]]
    end

    subgraph INF["Container: inferência (FastAPI / Flask)"]
        LOAD["Carrega model.joblib<br/>na inicialização"]
        HEALTH["GET /health"]
        PRED["POST /predict"]
        LOAD --> PRED
    end

    subgraph CLIENT["Aplicação cliente"]
        CLI["curl / script Python / UI simples"]
    end

    DS -- "download CSV" --> PRE
    FIT -- "joblib.dump()" --> MODEL
    EVAL --> META
    MODEL -- "volume montado<br/>(-v artefatos:/app/model)" --> LOAD
    META --> LOAD
    CLI -- "HTTP JSON (features)" --> PRED
    PRED -- "JSON (preço previsto)" --> CLI
    CLI -- "HTTP GET" --> HEALTH

```

```mermaid
sequenceDiagram
    participant C as Cliente
    participant API as Container de inferência
    participant M as model.joblib

    Note over API,M: Inicialização
    API->>M: joblib.load()
    M-->>API: modelo sklearn

    C->>API: GET /health
    API-->>C: 200 {"status":"ok"}

    C->>API: POST /predict {open, high, low, volume...}
    API->>API: valida entrada e monta o vetor de features
    API->>M: model.predict(X)
    M-->>API: valor previsto
    API-->>C: 200 {"prediction": 1234.56}


```
https://www.kaggle.com/datasets/varpit94/ethereum-data/data 
