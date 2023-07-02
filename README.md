Инструкции за използване
--

Този проект съдържа кода на дипломна работа на тема *Машинно самообучение с утвърждение за контрол на симулиран модел на роботизирана ръка*  
Видеа от прогресите на различните модели ще кача [тук](https://drive.google.com/drive/folders/1Ft0yp1Um0f3qD70XKgsWvFDyLHn5pmw-?usp=sharing) (към момента има само 1).   

### Предварителни изисквания
Този проект е разработен изцяло на операционната система `Ubuntu 22.04 (Jammy Jellyfish)`. Инструкциите са само за конфигурация при тази операционна система.  
Инсталиран `git`, `python (version 3.7.*)` (надолу има инструкции за инсталация на точно тази версия), `virtualenv`

### 1. Инсталация на MuJoCo
Следвайте [инструкциите за инсталация](https://github.com/deepmind/mujoco#installation), описани от създателите. Накратко трябва да се изтегли изпълнимият файл за MuJoCo, и да се разархивира в папка `/home/<user>/.mujoco`, където `<user>` е името на Линукс потребителя Ви.  
Аз съм използвала версия `210` на MuJoCo.

### 2. Клониране на проекта
В избрана от Вас директория, изпълнете:
```
git clone https://github.com/rayag/pick_place_robosuite.git
```

### 3. Създаване на виртуална среда
#### Инсталиране на `python-3.7`
Тази стъпка е нужна единствено, ако нямате налична версията `3.7`.  
За да проверите това (при стандарта инсталация на Ubuntu) изпълнете командата:
```
ls /usr/bin/ | grep python3.7
```
Ако изпълнимите файлове на `python` интерпретатора са Ви в друга папка, тогава проверете там.

В случай, че нямате инсталиран `python3.7`, изпълнете командите:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
```
```
python3.7 -m pip install package
```
```
sudo apt-get install python3.7-distutils
```
Би трябвало описаното до тук да е достатъчно като инсталация, но може да има нужда и от допълнителни пакети.

#### Създаване на виртуална среда
Влезте в директорията `pick_place_robosuite`.   
Изпълнете командата:
```
virtualenv --python="/usr/bin/python3.7" "./venv"
```
Разбира се, може да замените `/usr/bin/python3.7` с директорията, в която Ви се намират изпълнимите файлове на `python3.7` и `./venv` с друга директория, в която прецените, че искате да помещавате файловете, свързани с виртуалната среда, ако сте избрали друга директория, от сега нататък, в следващите команди би трябвало да я замените, там където съм използвала `./venv`.  
След изпълнение на горната команда, би трябвало в директорията да се е появила нова директория `venv`.


#### Активиране на виртуалната среда  
Изпълнете (в `pick_place_robosuite`):
```
source venv/bin/activate
```
Ако всичко е било наред, следва командата 
```
python --version
```
Да изведе `3.7.*`, където `*` е съответна микроверсия и няма значение, конкретно при мен е 15.   
В случай, че желаете да деактивирате средата, просто изпълнете 
```
deactivate
```

### 4. Инсталиране на нужните пакети
Изпълнете:
```
pip install -r requirements.txt
```
На този етап би трябвало да можете да изпълните епизод, в който агентът прави случай действия:
```
python environment/pick_place_wrapper.py
```
Ако видите грешка `ModuleNotFoundError: No module named '...'` изпълнете  `export PYTHONPATH=.:$PYTHONPATH` в основната директория на проекта, за съжаление все още не знам как да избягам от това.  

### 5. Стартиране на тренировка
Тук ще опиша случаите за Hindsight Experience Replay + DDPG.  
Основната част от кода се намира във файла `rl_agent/her_ddpg.py`. Параметрите могат да бъдат видяни с командата:
```
python rl_agent/her_ddpg.py --help
```
Което трябва да изведе:
```
usage: her_ddpg.py [-h] [-r RESULTS_DIR] [-chkp CHECKPOINT] [-alr ACTOR_LR]
                   [-clr CRITIC_LR] [--epochs EPOCHS]
                   [--it_per_epoch IT_PER_EPOCH] [--ep_per_it EP_PER_IT]
                   [--exp_eps EXP_EPS] [--normalize] [--update_it UPDATE_IT]
                   [--k K] [--horizon HORIZON] [--seed SEED] [--use_states]
                   [--start_from_middle] [-a {train,rollout}]
                   [-t {reach,pick,pick_and_place}] [--reach_pi REACH_PI]
                   [--dense_reward] [--descr DESCR]

optional arguments:
  -h, --help            show this help message and exit
  -r RESULTS_DIR, --results_dir RESULTS_DIR
                        Directory which will hold the results of the
                        experiments
  -chkp CHECKPOINT, --checkpoint CHECKPOINT
                        Checkpoint dir to load model from, should contain
                        *.pth files
  -alr ACTOR_LR, --actor-lr ACTOR_LR
  -clr CRITIC_LR, --critic-lr CRITIC_LR
  --epochs EPOCHS
  --it_per_epoch IT_PER_EPOCH
                        Number of iterations per epoch
  --ep_per_it EP_PER_IT
                        Number of episodes per iteration
  --exp_eps EXP_EPS     Exploration epsilon
  --normalize           If set, data will be normalized
  --update_it UPDATE_IT
                        Number of update iterations per epoch
  --k K                 k parameter of HER
  --horizon HORIZON     Timesteps included in one episode
  --seed SEED           Random seed
  --use_states          If true, the agent uses predefined states
  --start_from_middle   For pick_and_place agent, if true, starts episodes
                        with already picked object
  -a {train,rollout}, --action {train,rollout}
  -t {reach,pick,pick_and_place}, --task {reach,pick,pick_and_place}
  --reach_pi REACH_PI   Path to saved reach agent
  --dense_reward        if set, we use dense reward
  --descr DESCR         Description, appended to the directory name
```
Пример за трениране на задачата REACH, изучаване 20%, нормализация на данните и епизод с дължина 100
```
python rl_agent/her_ddpg.py -a train -t reach --exp_eps 0.2 --normalize --horizon 100
```
За да използвате генерираните начални състояния, трябва да изтеглите следната [папка](https://drive.google.com/drive/folders/1VmMiR0cpHTK58NRYqWgX51G24zJmC_N2?usp=sharing) в директорията на проекта.   
Това става с опцията `--use_states`
```
python rl_agent/her_ddpg.py -a train -t reach --exp_eps 0.2 --normalize --horizon 100 --use_states
```
При задачите PICK и PICK\_AND\_PLACE може да се използва трениране от състояние на вече захванат предмет, това става като се добави параметър `start_from_middle`:
```
python rl_agent/her_ddpg.py -a train -t reach --exp_eps 0.2 --normalize --horizon 100 --use_states --start_from_middle
```

Ако искате да използвате тренриане на няколко процеса, това става като се добави `mpirun -n ` и желаният брой на процесите пред командата. Например за трениране с 4 процеса:
```
mpirun -n 4 python rl_agent/her_ddpg.py -a train -t reach --exp_eps 0.2 --normalize --horizon 100
```

### 5. Изпробване на вече трениран модел
Тренирани модели са качени в следната [папка](https://drive.google.com/drive/folders/1al8HecsIZ3Vt4prc-_UkV8eLJX2VtXfF?usp=sharing) (качила съм само по един примерен, които би трябвало да работи добре, но от robosuite [не гарантират](https://robosuite.ai/docs/algorithms/demonstrations.html#warnings) еднаквото поведение дори и на събирани демонстрации, така че не мога да твърдя със сигурност, че ще работят).
Та например, ако сте разархивирали горепосочените модели в директория `trained_models`, тогава би трябвало да можете да изпълните следното:
```
python rl_agent/her_ddpg.py -a rollout --task reach --horizon 100 -chkp trained_models/reach_model/checkpoint/
```
За визулаизация на тренирането на този модел:
```
python vis/visualize -d trained_models/reach_model/
```

Описание на директориите
--
`environment` съдържа обвивката на robosuite средата, това е средата, над която агентът се обучава  
`experiments` съдържа описание на експерименти с *DDPG* и *DDPG с Prioritized Experience Replay* (но не и за *HER*)  
`logger` съдържа кода на компонент, който се занимава със записване на артефакти от тренирането, като прогрес, грешки и т.н.  
`networks` съдържа дефинициите на мрежите на актьора и критика  
`replay_buffer` съдържа кода на буферите за повторения (имаме обикновен, с приоритизация, с ретроспекция), допълнителните класове, нужни за тяхната дефиниция, и нормализатора   
`rl_agent` съдържа кода на агентите, има 4 агента: *DDPG*, *DDPG+PER*, *DDPG+HER*, *High-Level Choreograph*, кодът за специално *Reach* е изведен в друг файл заради добавена специфика, описана в дипломната работа. Кодът на High-Level хореографа НЕ Е параметризиран, надявам се в следващи версии да успея да го направя.     
`subtask_agents` това са агенти, изпълняващи подзадачи с вече тренирани модели.
`vis` съдържа код с визуализации, част от кода се отнася до диаграми, представени в дипломната работа, затова не е част от оснавната функционалност.  
`object_detection` съдържа кода на линейната регресия, съпоставяща ограничаваща кутия на реални координати.
`yolov7` е (с много малки промени) копие на https://github.com/WongKinYiu/yolov7


Допълнителни бележки
--
[Папка](https://drive.google.com/drive/folders/1VmMiR0cpHTK58NRYqWgX51G24zJmC_N2?usp=sharing) съдържа също така и изображенията, използвани за обучение на модела за обектно разпознаване.  

Има голяма вероятност обектното разпознаване да не сработи на вашите устройства, тъй като съм направила една промяна по кода на robosuite, която позволява да се използват наблюдения от камерите по време на изпълнение. Това не беше налично във версията, която използвах, но е налично в по-нови [версии](https://github.com/ARISE-Initiative/robosuite/issues/397).  

Папка `yolov7` е (с много малки промени) копие на https://github.com/WongKinYiu/yolov7.

Източници
--
- https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
- https://github.com/Howuhh/prioritized_experience_replay
- https://youtu.be/-QWxJ0j9EY8
- https://arshren.medium.com/step-by-step-guide-to-implementing-ddpg-reinforcement-learning-in-pytorch-9732f42faac9
- https://github.com/TianhongDai/hindsight-experience-replay