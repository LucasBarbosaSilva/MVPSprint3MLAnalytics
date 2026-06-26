<h1 align="center"> MVP - Sprint 3 - Machine Learning & Analytics</h1>
<h2 align="center">Pós-Graduação em Ciência de Dados e Analytics pela PUC RJ</h2>
<h3 align="center">Modelo de Classificação Utilizando o Dataset Bank Marketing (UCI Repository) </h3>

**Dataset:** [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**Colab MVP:** [Notebook Colab do MVP](https://colab.research.google.com/drive/1jUUd9x8TjlZmgaVYFEJKt5N0R5i9GFYc?usp=sharing)

## Descrição do Problema:
Consiste na elaboração de um modelo de classificação para a predição do resultado de uma campanha de marketing direto (chamadas telefônicas) de uma instituição bancária portuguesa. O objetivo da classificação é prever se o cliente irá se inscrever em um depósito a prazo (variável result).

## Fonte dos dados

O dataset [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)   é um dataset disponível no UCI Machine Learning Repository. Os dados referem-se a campanhas de marketing direto de uma instituição bancária portuguesa. As campanhas de marketing baseavam-se em chamadas telefónicas. Frequentemente, era necessário mais do que um contacto com o mesmo cliente para apurar se o produto (depósito a prazo bancário) seria contratado ("sim") ou não ("não"). O dataset original é composto por 45211 instâncias e 17 colunas.

No entanto, para este projeto, iremos utilizar uma versão do dataset semitratada que foi desenvolvida no MVP da Sprint 2 - Análise de Dados e Boas Práticas ([link do projeto](https://github.com/LucasBarbosaSilva/MVPSprint2AnalisePreProcessamento)). Nesta versão, uma limpeza inicial dos dados já foi realizada, tratando dados faltantes, outliers, dados duplicados e renomeando as colunas para nomes mais significantes. Além disso, o dataset original possui um alto desbalançeamento entre as duas classes alvo, e o projeto anterior tentou diminuir essa diferença aplicando algumas técnicas de deduplicação. Por fim, além das colunas originais do dataset, algumas colunas contínuas foram discretizadas em faixas, para facilitar a análise e modelagem do problema.

Desse modo, a versão do dataset utilizada ao longo deste MVP, se encontra no repositório do GitHub deste projeto ([link](https://github.com/LucasBarbosaSilva/MVPSprint3MLAnalytics/blob/main/dataset/bank-marketing-pre-processed.csv)). A base contém 30888 e 24 colunas (as originais do dataset e algumas criadas no pré-processamento).

## Dicionário de dados
| Coluna | Tipo | Descrição | Será usada? | Observações |
|---|---|---|---|---|
| client_age | numérica | Idade dos clientes contactados | não | Será utilizada a coluna com os dados discretizados client_age_group |
| client_job | categórica | Ocupação dos clientes contactados | sim | Valores possíveis: admin., blue-collar, entrepneur housemaid, management, <br> retired, self-employed, services, student, technician, unemployed, unknown |
| client_marital_status | categórica | Situação conjugal dos clientes contactados | sim |  Valores possíveis: maried, single, divorced* unknown |
| client_education_level | categórica | Nível de educação dos clientes contactados | sim | Valores possíveis: primary, secondary, tertiary, unknown |
| client_financial_default | binária | Informa se o cliente já entrou em incumprimento financeiro no passado. | sim | Valores possíveis: yes/no |
| client_average_annual_balance | numérica | Saldo médio anual, em euros | não | Ao invés desta coluna iremos utilizar a coluna client_average_annual_balance_group. |
| client_housing_loan | binária | Informa se o cliente possui empréstimo imobiário. | sim | yes/no |
| client_personal_loan | binária | Informa se o cliente possui empréstimo pessoal | sim | yes/no |
| contact_type | categórica | Tipo de contato durante a campanha | sim | Valores possíveis: cellular, telephone |
| contact_day_of_month | numérica | Dia do último contato | não | Ao invés desta coluna iremos utilizar a coluna contact_week. |
| contact_month | categórica | Mês do último contato. | sim | Valores possíveis: jan, feb, mar, ... oct, nov, dec |
| contact_duration | numérica | Duração do último contato, em segundos | não | Ao invés desta coluna iremos utilizar a coluna contact_duration_group. |
| number_contacts_campaign | numérica | Número de contatos realizados durante esta campanha para este cliente (inclui o último contato). | não | Ao invés desta coluna iremos utilizar a coluna number_contacts_campaign_group. |
| days_last_contact | numérica | Dias de folga: número de dias decorridos desde o último contato com o  cliente em uma <br> campanha anterior.  | não | Ao invés desta coluna iremos utilizar a coluna days_last_contact_group. |
| number_contacts_previous_campaign | numérica | Número de contatos realizados antes desta campanha para  este cliente. | não | Ao invés desta coluna iremos utilizar a coluna days_last_contact_group. |
| result_previous_campaign | categórica | Resultado da campanha de marketing anterior. | sim | Valores possíveis: sucess, failuer, other. |
| client_age_group | categórica | Discretização por faixa etária da coluna client_age.  | sim | Valores possíveis: Jovem, Adulto, Meia-idade, Idoso. |
| contact_week | categórica | Discretização em semanas da variável contact_day_of_month. | sim | Valores possíveis: week 1, week 2, week 3, week 4. |
| client_average_annual_balance_group | categórica | Discretização por frequência da variável client_average_annual_balance | sim | Valores possíveis: negative, (-0.001, 146.0], (146.0, 533.0], (533.0, 1536.25], (1536.25, 13014.0]**|
| contact_duration_group | categórica | Discretização por frequência da variável contact_duration. | sim | Valores possíveis: (59.999, 124.0], (124.0, 200.0], (200.0, 338.0], (338.0, 1419.0]** |
| number_contacts_campaign_group | categórica | Discretização por frequência da variável number_contacts_campaign. | sim | Valores possíveis: (0, 1], (1.999, 3.0], (3.0, 4.0],  (4.0, 16.0]**|
| days_last_contact_group | categórica | Discretização por frequência da variável days_last_contact. | sim | Valores possíveis: never_before, (0.999, 126.0], (126.0, 190.0], (190.0, 321.5], (321.5, 616.0]** |
| number_contacts_previous_campaign_group | categórica | Discretização por frequência da variável number_contacts_previous_campaign. | sim | Valores possíveis: not_contacted, (0.999, 2.0], (2.0, 4.0], (4.0, 20.0]** |
| result_campaign | alvo | Resultado da campanha: indica se o cliente subscreveu um depósito prazo. | sim | yes/no |

\* Nota 1: divorcidado refere-se tanto a pessoas divorciadas, quanto a pessoas viúvas.

** Nota 2: A notação (x, y] indica que o início do intervalo é exclusivo e o final é inclusivo.

## Síntese da análise exploratória

Os principais pontos observados durante a análise exploratória foram:
- Utilizar a validação cruzada, para compensar o desbalanceamento do data set;
- Codificar os valores yes/no em símbolos binários 1/0;
- Não existem valores ausentes ou valores inválidos para as variaáveis categóricas;
- A variável contact_duration_group não poderá ser utilizada durante o treinamento, pois traz uma informação posterior ao contato;
- As demais features permanecem no dataset até que testes com diferentes combinações de colunas mostrem um conjunto mais efetivo;
- Será necessário codificar as colunas utilizando os codificadores ordinais e não ordinais;

## Resultados iniciais:
Os seguintes modelos foram treinados e testados em suas versões mais básicas:

| modelo | accuracy	| precision | recall | f1_weighted | roc_auc |	train_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.842977 | 0.000000 |	0.000000 | 0.771154 |	0.500000 |	1.256 |
| KNeighborsClassifier | 0.845150 | 0.515546 |	0.248907 | 0.821761 |	0.699211 |	8.763 |
| DecisionTreeClassifier | 0.769622 | 0.292447 |	0.329894 | 0.775062 |	0.592384 |	1.350 |
| GaussianNB | 0.770640 | 0.330987 |	0.451540 | 0.784242 |	0.718822 |	10.041 |
| VotingClassifier | 0.831969 | 0.453246 |	0.333134 | 0.821220 |	0.726344 |	9.910 |
| RandomForestClassifier | 0.838675 | 0.473261 |	0.237703 | 0.815556 |	0.728991 |	24.933 |
| ExtraTreesClassifier | 0.825401 | 0.407183 |	0.245651 | 0.806903 |	0.670472 |	32.154 |
| GradientBoostingClassifier | 0.860228 | 0.662293 |	0.225913 | 0.829969 |	0.790553 |	27.467 |
| AdaBoostClassifier | 0.857823 | 0.669517 |	0.186746 | 0.822170 |	0.766896 |	17.762 |
| XGBClassifier | 0.855280 | 0.577453 |	0.294262 | 0.834956 |	0.779687 |	4.223 |

## Análise dos resultados iniciais
Os cinco modelos com melhor **acurácia** foram:
1. GradientBoostingClassifier (0.860228)
2. AdaBoostClassifier (0.857823)
3. XGBClassifier (0.855280)
4. KNeighborsClassifier (0.845150)
5. baseline (0.842977)

Os cinco modelos com a maior taxa de **recall** foram:

1. GaussianNB (0.451540)
2. VotingClassifier (0.333134)
3. DecisionTreeClassifier (0.329894)
4. XGBClassifier (0.294262)
5. KNeighborsClassifier (0.248907)

Os cinco modelos com melhor métrica  **ROC** foram:
1. GradientBoostingClassifier (0.790553)
2. XGBClassifier (0.779687)
3. AdaBoostClassifier (0.766896)
4. RandomForestClassifier (0.728991)
5. VotingClassifier (0.726344)

A tabela a seguir mostra um resumo dos modelos que se destacaram nas métricas a cima. Além do indicador se estava ou não em uma das listas acima, foi adicionado o tempo de treinamento. Os modelos foram ordenados pela quantidade de indicadores e pelo menor tempo de treinamento:

| modelo | melhor acurácia | melhor recall | melhor roc_auc | tempo de treinamento |
| --- | --- | --- | --- | --- |
| XGBClassifier | ✅ | ✅ | ✅  | 4.223 |
| KNeighborsClassifier | ✅ | ✅ | ❌ | 8.763 |
| VotingClassifier | ❌ | ✅ | ✅ | 10.041 |
| GradientBoostingClassifier | ✅ | ❌ | ✅ | 27.467 |
| baseline | ✅ | ❌ | ❌ | 1.256 |
| GaussianNB | ❌ | ✅ | ❌ | 1.350 |
| DecisionTreeClassifier | ❌ | ✅ | ❌ | 2.329 |
| AdaBoostClassifier | ✅ | ❌ | ❌ | 16.303 |
| RandomForestClassifier | ❌ | ❌ | ✅ | 24.933 |
 
Desse modo, o modelo que seguiremos para a etapa de finetuning é o **XGBClassifier**.

## Resultados após otimização:

Para o modelo XGBClassifier os seguintes hiperparâmetros foram testados:

| hiperparâmetro | descrição | valores de teste |
| --- | --- | --- |
| n_estimators | Número total de árvores a construir. O padrão é 100. | Valores aleatórios entre 50 e 300 |
| max_depth | Profundidade máxima de uma árvore. As árvores mais profundas <br> captam padrões mais complexos, mas aumentam o risco de *overfitting*. O valor padrão é 3. | Valores aleatórios entre 3 e 10 |
| learning_rate | Ou `eta`. Ela escala a contribuição de cada nova árvore. O valor predefinido é 0,1. | [0.01, 0.05, 0.1, 0.2] |
| subsample | Razão de subamostragem das instâncias de treinamento. Definir esse valor como 0,8 <br> significa que o XGBoost seleciona aleatoriamente 80% das linhas de dados para construir as árvores. O padrão é 1. | [0.5, 0.6, 0.8, 1.0] |
| colsample_bytree | Proporção de subamostragem de colunas (recursos) ao construir cada árvore. O padrão é 1. | [0.6, 0.8, 1.0] |
| scale_pos_weight | Controla o equilíbrio entre pesos positivos e negativos. Muito útil para lidar com conjuntos de dados desbalanceados. | [1, (y.value_counts()[0] / y.value_counts()[1])] |

Após realizar um finetuning das principais variáveis do modelo XGBClassifier os melhores valores para os parâmetros encontrados:
```
{
 'colsample_bytree': 0.8,
 'learning_rate': 0.05,
 'max_depth': 6,
 'n_estimators': 108,
 'scale_pos_weight': np.float64(5.368659793814433),
 'subsample': 0.5
}
```

E obtivemos os seguintes resultados:

| modelo |	accuracy |	precision |	recall |	f1_weighted |	test_roc_auc |	train_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 	0.842977  |	0.000000 |	0.000000 |	0.771154 |	0.500000 |	1.267 |
| XGBClassifier |	0.855280 |	0.577453 |	0.294262 |	0.834956 |	0.779687 |	6.330 |
| XGBClassifier_tun |	0.779058 |	0.381445 |	0.655088 |	0.800300 |	0.793341 |	4.812 |

Apesar de ter reduzido a acurácia, o modelo otimizado melhorou consideravelmente a métrica test_recall (+0,36), sem haver uma redução significativa o test_f1_weighted (-0,3). Além disso, ainda houve um aumento na test_roc_auc (+0,02) com uma leve redução do tempo de treinamento.

## Conclusão
O objetivo deste modelo era realizar a predição do comportamento de um cliente ao ser abordado por uma campanha de marketing. Diversos modelos foram analisados a fim de encontrar o que melhor se adapte ao problema proposto.

Durante o treinamento, uma limitação encontrada foi o grande desbalanceamento entre as classes. Cerca de 80% dos dados correspondem a classe 0, o que requer estratégias mais elaboradas de treinamento. Mesmo com essas limitações, alcançamos bons valores de métricas, alcançando mais de 60% de recall e quase 80% de poder de previsão avaliado pela curva roc.

Inicialmente, achavasse que a métrica de acurácia seria uma boa métrica para avaliar o treinamento, mas, no decorrer do projeto, percebeu-se que a acurácia não traduzia o real estado do modelo. Uma prova disso foi o resultado das métricas iniciais, em que a acurácia do modelo baseline estava entre os melhores valores, mas o modelo tinha precisão e recall zerados, indicando que o modelo não identifica nenhum dos clientes que aceitaram a campanha.

Desse modo, como métricas de avaliação, optou-se por, além da curca roc, avaliar também o recall. A curva roc já estava prevista nas métricas iniciais do projeto e foi utilizada para avaliar a capacidade preditiva do modelo, sendo 0,5 um modelo com nenhuma predição (como o baseline) e 1 um modelo excelente. Já o recall foi preferido a precisão, pois, para o tipo de problema escolhido, era preferível ter a maior certeza sobre os clientes que aceitariam a campanha, mesmo abordando alguns que não aceitariam, do que aacertar o máximo de predições (positivas ou negativas), avaliando a precisão.

A comparação do modelo baseline com o modelo final pode ser vista na tabela abaixo:

| Modelo | acurácia | recall | roc_auc | Tempo de treino |
|---:|---:|---:|---:|---:|
| Baseline | 0.842977 | 0.000000 | 0.500000 | 1.267 |
| XGBClassifier_tun | 0.779058 | 0.655088 | 0.793341 | 4.812 |

Assim, por mais que o modelo final tenha conseguido uma acurácia menor do que o modelo baseline, ele superou, e muito, as demais métricas (recall e roc_auc), mostrando-se um bom preditor e com o tempo de treinamento muito abaixo do estipulado nos critérios de sucesso iniciais (menos de 30 segundos).

Por fim, como próximos passos, poderíamos buscar treinar o modelo com mais dados positivos, para diminuir a diferença entre as classes, ou buscar encontrar um modelo com uma precisão mais alta, levando em consideração que talvez não seja interessante incomodar clientes que não aceitariam o contrato. Além disso, poderíamos testar diferentes combinações de colunas, inclusive, trazendo as colunas contínuas de volta, a fim de comparar resultados.
