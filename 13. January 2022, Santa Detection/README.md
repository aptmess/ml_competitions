# Описание задачи
Мы предлагаем вам задачу на классификацию изображений.
В предоставленном датасете есть набор картинок, на которых есть или Дед Мороз, или Санта Клаус или вообще нет.
Мы предлагаем вам написать модель, которая будет присваивать изображению одну из трёх категорий:

- Есть Дед Мороз **( class_id ==1 )**
- Есть Санта Клаус **( class_id == 2 )**
- Никого из персонажей нет **( class_id == 0 )**

Мы подготовили датасет из 1280 изображений, которые вы можете использовать для обучения своих моделей и локальной проверки решений. Архив с изображениями и train.csv вы можете скачать по ссылкам:
- [архив с изображениями](https://contestfiles.s3.eu-central-1.amazonaws.com/nyds/train.zip)
- [train.csv](https://contestfiles.s3.eu-central-1.amazonaws.com/nyds/train.csv)

# Работа с репозиторием
Для вас подготовлен шаблон репозитория над проектом. 
В нём вы сможете найти файлы:
- train.py - где мы ожидаем логику обучения модели
- run.py - где мы ожидаем логику расчета модели и генерации submission.csv  файла с предсказанием.

и директории:
- data/out - сюда нужно складывать файл с предсказанием submission.csv
- data/test - сюда при проверке модели монтируются изображения из тестового датасета
- data/weight - здесь вы можете хранить веса модели.

так же, в корне репозитория есть Dockerfile, который запускается при каждом git push и проверке модели.

После каждой **git push** в **main** ветку, из неё собирается docker image.
При нажатии на "Проверить модель" (на странице с описанием задачи), к собранному после git push образу монтируется папка с тестовыми изображениями и запускается выполнение программы. 
В результате выполнения программы, в папке data/out должен появиться файл с предсказаниями (submission.csv), который далее расчитывается по метрике f1 weighted.

После успешного расчета вы получите email уведомление на почту с результатом расчета на публичной части тестовых данных. Так же, информация обо всех проверках будет накапливаться во вкладке "Предыдущие решения" на странице с описанием задачи.

После окончания приема решений, лидерборд будет пересчитан на отложенной части тестовых данных и по обновлённому лидерборду будут выявлены призёры соревнования.

**Вы не можете отправлять на проверку коммиты чаще, чем раз в 15 минут**

**Время выполнения вашей программы после сборки не должно превышать 60 минут**

**Обратите внимание, что среда, в которой производится выполнение расчета оторвана от интернета. Все необходимые зависимости следует устанавливать на этапе сборки приложения**

**Использование сторонних данных запрещено. Все призовые места будут нами провалидированы в ручную.  Модели могут быть предобучены только на ImageNet**