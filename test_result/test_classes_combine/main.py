def classes_matchingT(target_classes_all, triggered_classes_all):
    start_index = len(target_classes_all)
    for i in range(len(triggered_classes_all)):
        triggered_classes = triggered_classes_all[i]
        target_classes = target_classes_all[i]

        for triggered_class in range(triggered_classes.size(0)):
            target_classes_all.append(target_classes)
            triggered_classes_all.append(triggered_classes[triggered_class])

    end_index = len(target_classes_all)

    if start_index != end_index:
        target_classes_all = target_classes_all[start_index:]
        triggered_classes_all = triggered_classes_all[start_index:]

    return target_classes_all, triggered_classes_all


if __name__ == '__main__':
    for file in os.listdir('D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip'):
        FILE_ROOT_PATH = os.path.join('D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip', file)
        MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
        DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')
        StartTime = time.time()
        setup_seed(SEED)
        model = load_model(MODEL_PATH)
        print(f"{'-' * 20}{file}{'-' * 20}")
        # pre_screening會回傳過濾後可疑的target classes與victim classes
        FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)
        FilteredTargetClasses_T, FilteredVictimClasses_T = Pre_Screening_T(model, DATA_PATH, num_classes=5)


        print(f'FilteredTargetClasses: {FilteredTargetClasses}, FilteredVictimClasses: {FilteredVictimClasses}')
        print(f'FilteredTargetClasses: {FilteredTargetClasses_T}, FilteredVictimClasses: {FilteredVictimClasses_T}')

        assert FilteredTargetClasses == FilteredTargetClasses_T
        #print(FilteredVictimClasses == FilteredVictimClasses_T)
        #assert int(torch.all(FilteredVictimClasses == FilteredVictimClasses_T))

        if FilteredVictimClasses_T is None:
            if FilteredTargetClasses_T is None:
                number_of_classes = 0
                print(None)
            else:
                # universal backdoor(因為pre-screening的輸出假如只有target label)
                number_of_classes = 1
                print('universal')
        else:
            # label specific backdoor
            print('label spec')
            x = classes_matchingT(FilteredTargetClasses_T, FilteredVictimClasses_T)
            print(f'paper: {x}')
        if FilteredVictimClasses is None:
            if FilteredTargetClasses is None:
                number_of_classes = 0
                print('安全')
            else:
                # universal backdoor(因為pre-screening的輸出假如只有target label)
                number_of_classes = 1
                print('universal')
        else:
            # label specific backdoor
            print('label spec')
            y = classes_matchingT(FilteredTargetClasses, FilteredVictimClasses)
            print(f'me: {y}')
            assert x == y


        TimeCost = time.time() - StartTime
        print(f"{TimeCost}")