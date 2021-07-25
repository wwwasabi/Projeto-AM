# Projeto-AM

### Considere os dados "Yeast Data Set":<br>
#### cin IN1102

a) Use validação cruzada estratificada “5-folds” para avaliar e comparar os classificadores descritos 
abaixo. Quando necessario, retire do conjunto de aprendizagem, um conjunto de validação (20%)
para fazer ajuste de hiper-parametros e depois treine o modelo novamente com o conjunto
aprendizagem + validação. Use amostragem estratificada.<br>

b) Obtenha uma estimativa pontual e um intervalo de confiança para cada metrica de avaliaa ̧ ão do
classificador (Taxa de erro, precisão, cobertura, F-measure);<br>

c) Usar o Friedman test (teste não parametrico) para comparar os classificadores, e o pós teste
(Nemenyi test)<br>

![questao2](https://user-images.githubusercontent.com/35909969/121105595-c6f9f500-c7da-11eb-9a1c-9924fc5ca3e2.PNG)<br>

ii) Treine um classificador bayesiano baseados em k-vizinhos. Use a distância Euclidiana para definir a
vizinhança. Use conjunto de validação para fixar o o número de vizinhos k.<br>

iii) Treine um classificador bayesiano baseado em janela de Parzen. Use a função de kernel multivariada
produto com o mesmo h para todas as dimensões e a função de kernel Gaussiana unidimensional. Use
conjunto de validação para fixar o parâmetro h.<br>

iv) Treine um classificador baseado em regressão logistica para cada classe e use a bordagem “um contra
todos’ para classificar os exemplos. <br>

v) Treine um classificador usando a regra do voto majoritario a partir dos classificadores i) a iv).
