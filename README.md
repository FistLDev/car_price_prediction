# car_price_prediction

В данном проекте был реализован разведовательный анализ даннх с характеристиками машин и целевой переменной -
ценой на данные машины. Создано и обучено несколько моделей линейной регрессии 
и проведен подсчет метрик качества. А также написан сервис на FastApi, повзоляющим получать прогнозированную цену 
на автомобиль.

 ## Результаты
По итогу, удалось добиться значения по метрике R^2 в 0.63.
Этот результат считается лучшим и был получен, благодаря кодировке категориальных фич при помощи OneHotEncoding.

К сожалению, в реализации сервиса не удалось выдавать прогнозированную цену, основываясь на предсказаниях модели с закодированными
категориальными признаками.

Screencast работы сервиса: https://disk.yandex.ru/i/Ox1JFI4WbkO4iQ