# Домашнее задание 1
#### EDA
Был проведен первичный анализ данных:
* Исследованы основные статистики вещественных и категориальных признаков для трейна и теста
* Исследованы пропуски в обоих датасетах. Признаки, которые имеют пропуски: mileage, engine, max_power, torque, seats
* Были удалены дубли из трейна
* У признаков mileage, engine, max_power были удалены единцы измерения
* Также были заполнены пропущенные значения медианами трейна
* Были постороены pairplot-ы всех числовых признаков, а также heatmap

Выводы:
* Слабая корреляция между фичами max_power / km_driven и engine / year
* Сильная корреляция между фичами max_power / engine и seats / engine

#### Обучение моделей
##### 1. Классическая линейная регрессия с дефолтными параметрами только на вещественных признаках
Качество модели оценивалось с помощью метрик R2 и MSE.
* R2_score for train: 0.5932097759804064
MSE for train: 116601673873.20876

* R2_score for test: 0.5946576430490256
MSE for test: 233002361585.21136

Результат на трейне и тесте оказался одинаковый. Значимось признаков была оценена по весам модели.
Признаки в порядке убывания значимости:
* max_power
* year
* engine
* km_driven
* mileage
* seats

##### 2 Lasso-регрессия только на вещественных признаках
###### C дефолтными параметрами и стандартизацией
Качество обученной модели не отличается от классичесой регрессии:
* R2_score for train: 0.593209775945082
MSE for train: 116601673883.33408

* R2_score for test: 0.5946564769596101
MSE for test: 233003031886.7058

Веса не занулились, значит при дефолтном значении параметра alpha все признаки являются значимыми для предсказания.

###### C подобранными параметрами
Наилучшие параметры подбирались с помощью GridSearchCV.
Вес одного из признаков занулился, однако качество обученной модели на тесте стало хуже. 
Лучший коэффициент регуляризации: 9991
Вес признака "seats" занулился
* R2_score for train: 0.5902890392302964
MSE for train: 117438869002.04109

* R2_score for test: 0.5823199447865226
MSE for test: 240094422857.3489

##### 3 ElasticNet с подобранными параметрами
Наилучшим параметром оказался l1_ratio = 0.9. Это говорит, о том, что лучше использовать Ridge регрессию.

Качество обученной модели стало еще хуже.
* R2_score for train: 0.5892558898114422
MSE for train: 117735009234.73686

* R2_score for test: 0.5727814580656632
MSE for test: 245577417402.07706

##### 4 Ridge-регрессия с подобранными параметрами и категориальными фичами
Для использования категориальных признаков в линейной регрессии был применен метод OneHotEncoding. А также применена стандартизация признаков для лучшей работы регуляризации.

Данная модель показала наилучшее качество, однако заметно улучшить качество не удалось.
* R2_score for train: 0.6439656277240604
MSE for test: 102053100867.46852

* R2_score for test: 0.6013905862818958
MSE for test: 229132073551.465

#### Feature Engineering
Для еще большего улучшения качества были добавлены следующие столбцы:
1. Отношение числа "лошадей" к объему двигателя
2. Год в квадрате, так как зависимость цены от года похожа на  квадратичную
3. Сколько км в среднем автомобиль проезжал за год
4. Пороговый признак, если владелец третий или больше

#### Обучение модели после добавления новых признаков
Было принято решение обучать Ridge-регрессию, так как на ней получился наилучший результат.
Модель показала еще более высокий результат.
* R2_score for train: 0.6592853452675391
MSE for test: 97661882486.69362

* R2_score for test: 0.6323014558206812
MSE for test: 211363623060.95657

#### Кастомная метрика
Необходимо было рассчитать среди всех предсказанных цен на авто  долю предиктов, отличающихся от реальных цен на эти авто не более чем на 10% (в одну или другую сторону).
Была реализована функция для рассчета данной бизнесовой метрики.
Полученный результат: 0.245
Качество модели оставляет желать лучшего :(

#### FastAPI
Далее при помощи FastAPI, был реализован сервис с простейшим функционалом с возможностью расчета цены для одного автомобиля или csv-файла с несколькими авто.
* [prediction for one object](https://github.com/maxgalanov/ml_hw1/blob/main/Screenshots%20Fast%20API/prediction_for_one_object.png)
* [prediction for csv file](https://github.com/maxgalanov/ml_hw1/blob/main/Screenshots%20Fast%20API/download_csv.png)

#### Возможные доработки и улучшения
Большинство идей не удалось  реализовать по причине нехватки времени и других дедлайнов :)
* Можно было не дропать признак  со значением крутящего момента, а распарсить его в два столбца.
* Добавить попарные произведения признаков в качестве новых фичей.
* Спарсить по названию автомобиля его класс, думаю это бы дало буст в качестве.
* Сделать интерфейс для сервиса, с возможностью загрузки, сохранения, просмотра расчетов.