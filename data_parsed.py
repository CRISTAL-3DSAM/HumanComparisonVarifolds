###
#Objects used for CVSSP3D artificial dataset, persons : individuals, ext_mouvs: motion names
persons = ['Adrian', 'Alan', 'Dave', 'EngJon', 'Graham', 'Jez', 'Jigna',
           'Joel', 'Marc', 'PengWei', 'Pete', 'Pip', 'Venura', 'Yacob']
ext_mouvs = ['faint', 'fastrun', 'fastwalk', 'rocknroll', 'runcircleleft', 'runcircleright', 'runturnleft',
             'runturnright', 'shotarm', 'slorun', 'slowalk', 'sneak', 'sprint', 'vogue', 'walkcircleleft',
             'walkcircleright', 'walkcool', 'walkcowboy', 'walkdainty', 'walkelderly', 'walkmacho', 'walkmarch',
             'walkmickey', 'walksexy', 'walktired', 'walktoddler', 'walkturnleft', 'walkturnright']
###
#Objects used for CVSSP3D real dataset, pou : individuals, the_dict: numbers of the motions, classes, id (number) of motion
pou = {1: ["Nikos", "Natali", "Chris", "Jon"], 2: ["Haidi", "Jean", "Hansung", "Joe"]}

the_dict = {"Walk": {1: [2, 15, 28, 41], 2: [1, 14, 29, 52]},
            "Run": {1: [3, 16, 29, 42], 2: [2, 15, 42, 30]},
            "Jump": {1: [4, 17, 30, 43], 2: [3, 16, 43, 31]},
            "Bend": {1: [5, 18, 31, 44], 2: [4, 17, 44, 32]},
            "Hand - wave": {1: [6, 19, 32, 45], 2: [5, 18, 45, 33]},
            "Jump in place": {1: [7, 20, 33, 46], 2: [6, 19, 46, 34]},
            "Sit - stand up": {1: [8, 21, 34, 47], 2: [7, 20, 47, 35]},
            "Run - fall": {1: [9, 22, 35, 48], 2: [8, 21, 48, 36]},
            "Walk - sit": {1: [10, 23, 36, 49], 2: [9, 22, 49, 37]},
            "Run - jump - walk": {1: [11, 24, 37, 50], 2: [10, 23, 50, 38]}}
            #"Handshake": {1: [13, 26, 39, 52], 2: [12, 25, 27, 40]}, # Interaction motions are discarded for experiments
            #"Pull": {1: [14, 27, 40, 53], 2: [13, 26, 28, 41]}}
classes = {}
i = 0
for key in the_dict:
    classes[key] = i
    i += 1


###
# Objects used for dyna dataset.
sids = ['50004', '50020', '50021', '50022', '50025',
            '50002', '50007', '50009', '50026', '50027']
pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
            'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
            'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
            'one_leg_jump', 'running_on_spot']