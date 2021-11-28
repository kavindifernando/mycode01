'''
CS 550: Machine Learning
Homework 3 - Genetic algorithm based cost sensitive multiclass classifier
Muhammed Cavusoglu, 21400653
'''


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from random import randint, random

def get_selected_features(selected_features, all_features):
    selected_f = []
    for sample in all_features:
        s_f = []
        for i in range(len(selected_features)):
            if selected_features[i] == 1:
                s_f.append(sample[i])
        selected_f.append(s_f)
    return selected_f

def get_predicted_labels(class_prob):
    predicted_labels = []
    
    for p in class_prob:
        predicted_labels.append(p.argmax() + 1)
            
    return predicted_labels
    
def get_class_miss_percentages(true_labels, predicted_labels):
    c1_missed = 0
    c1_tot = 0
    c2_missed = 0
    c2_tot = 0
    c3_missed = 0
    c3_tot = 0
    
    for i in range(len(true_labels)):
        if true_labels[i] == 1:
            c1_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c1_missed += 1
        
        if true_labels[i] == 2:
            c2_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c2_missed += 1
                
        if true_labels[i] == 3:
            c3_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c3_missed += 1
        
    c1_miss_percent = (100.00 * c1_missed) / c1_tot  
    c2_miss_percent = (100.00 * c2_missed) / c2_tot
    c3_miss_percent = (100.00 * c3_missed) / c3_tot
    
    if c1_miss_percent <= 1:
        c1_miss_percent = 1
    if c2_miss_percent <= 1:
        c2_miss_percent = 1
    if c3_miss_percent <= 1:
        c3_miss_percent = 1
        
    print( "Missed samples for each class: ", c1_missed, c2_missed, c3_missed)
    
    return c1_miss_percent, c2_miss_percent, c3_miss_percent
    
def load_data():
    train = open("ann-train.data", "r")
    training_data = []
    for line in train:
        training_data.append(line.strip().split(" "))
    
    convert_types(training_data)
    
    test = open("ann-test.data", "r")
    test_data = []
    for line in test:
        test_data.append(line.strip().split(" "))
    
    convert_types(test_data)
    
    training_features = []
    training_labels = []
    for sample in training_data:
        training_features.append(sample[:-1])
        training_labels.append(sample[-1])
        
    test_features = []
    test_labels = []
    for sample in test_data:
        test_features.append(sample[:-1])
        test_labels.append(sample[-1])
        
    cost = open("ann-thyroid.cost", "r")
    costs = []
    for line in cost:
        costs.append(float(line.strip().split(":")[1]))
    costs.append(0) # 21st feature is a comb. of 19th and 20th features
        
    return training_features, training_labels, test_features, test_labels, costs
    
def convert_types(data):
    for row in data:
        row[0] = float(row[0].strip())
        row[1] = int(row[1].strip())
        row[2] = int(row[2].strip())
        row[3] = int(row[3].strip())
        row[4] = int(row[4].strip())
        row[5] = int(row[5].strip())
        row[6] = int(row[6].strip())
        row[7] = int(row[7].strip())
        row[8] = int(row[8].strip())
        row[9] = int(row[9].strip())
        row[10] = int(row[10].strip())
        row[11] = int(row[11].strip())
        row[12] = int(row[12].strip())
        row[13] = int(row[13].strip())
        row[14] = int(row[14].strip())
        row[15] = int(row[15].strip())
        row[16] = float(row[16].strip())
        row[17] = float(row[17].strip())
        row[18] = float(row[18].strip())
        row[19] = float(row[19].strip())
        row[20] = float(row[20].strip())
        row[21] = int(row[21].strip())

def main():
    training_features, training_labels, test_features, test_labels, costs = load_data()
    data = [training_features, training_labels, test_features, test_labels, costs]
    clf = tree.DecisionTreeClassifier()
    
    pop = population(20, training_features[0])
    
    fitness_history = []
    for i in range(100):
        pop = evolve(pop, clf, data)
        pop_fitness = avg_fitness(pop, clf, data)
        fitness_history.append(pop_fitness)
        
        if pop_fitness < 250:
            break

    print ("Avg fitness history: ", fitness_history)
    
    # fittest_results(clf, training_features, training_labels, test_features, test_labels, costs)
    
def individual(length):
    # create an individual, which is binary repr. of selected features
    selected_features = [0] * len(length)
    
    for i in range(len(selected_features)):
        selected_features[i] = randint(0, 1)
        
    # 21st feature is a comb. of 19th and 20th features
    if selected_features[20] == 1: 
        selected_features[18] = 1
        selected_features[19] = 1
        
    return selected_features
    
def population(count, length):
    # create 'count' number of individuals
    return [individual(length) for _ in range(count)]

def fitness(individual, clf, data):
    # determine the fitness of an individual
    # misclassified class 1 (%) * misclassified class 2 (%) * misclassified class 3 (%) * feature selection cost (resulting value is converted to int)
    # lower is better
    print ("\n######################################################################")
    print ("Individual: ", individual)
    fs_cost = feature_selection_cost(individual, data[4])
    
    selected_train_f = get_selected_features(individual, data[0])
    selected_test_f = get_selected_features(individual, data[2])

    clf = clf.fit(selected_train_f, data[1])
    
    class_prob = clf.predict_proba(selected_test_f)
    predicted_labels = get_predicted_labels(class_prob)
    target_names = ['class 1', 'class 2', 'class 3']
    
    print(classification_report(data[3], predicted_labels, target_names=target_names))
    print( "Mean accuracy: ", clf.score(selected_test_f, data[3]))
    print ("No of correctly classified samples: ", accuracy_score(data[3], predicted_labels, normalize=False))
    
    c1_miss_percent, c2_miss_percent, c3_miss_percent = get_class_miss_percentages(data[3], predicted_labels)
    f_result = int(c1_miss_percent * c2_miss_percent * c3_miss_percent * fs_cost)
    
    print ("\nClass accuracies: \n", "class 1: ", (100 - c1_miss_percent), "%\nclass 2: ", (100 - c2_miss_percent), "%\nclass 3: ", (100 - c3_miss_percent), "%\n")
    print ("Feature selection cost: ", fs_cost)
    print ("Fitness: ", f_result)
    print ("######################################################################\n")
    
    return f_result

def avg_fitness(pop, clf, data):
    # average fitness of a population
    tot_fitness = 0
    for i in pop:
        tot_fitness += fitness(i, clf, data)
    return tot_fitness / len(pop)
    
def evolve(pop, clf, data, retain_percentage=0.50, random_select=0.05, mutate_prob=0.01):
    f_values = [(fitness(i, clf, data), i) for i in pop]
    individuals = [i[1] for i in sorted(f_values)]
    retain_length = int(len(pop) * retain_percentage)
    parents = individuals[:retain_length]
    
    # randomly add other individuals to increase diversity
    for i in individuals[retain_length:]:
        if random_select > random():
            parents.append(i)
            
    # mutate
    for i in parents:
        if mutate_prob > random():
            index_to_mutate = randint(0, len(i) - 1)
            i[index_to_mutate] = randint(0, 1)
            
            # make sure the result is still valid
            if i[20] == 1: 
                i[18] = 1
                i[19] = 1
    
    # crossover
    no_of_parents = len(parents)
    remaining_no_of_ind = len(pop) - no_of_parents
    children = []
    
    while len(children) < remaining_no_of_ind:
        male_index = randint(0, no_of_parents - 1)
        female_index = randint(0, no_of_parents - 1)
        
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)
    
    parents.extend(children)
    return parents
    
def feature_selection_cost(selected_features, costs):
    total_cost = 0
    
    for i in range(len(selected_features)):
        if selected_features[i] == 1:
            total_cost += costs[i]
            
    return total_cost
    
def fittest_results(clf, training_features, training_labels, test_features, test_labels, costs):
    # obtained from genetic algorithm runs
    fittest = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]
    
    print ("Test accuracies")
    test_acc_data = [training_features, training_labels, test_features, test_labels, costs]
    fitness(fittest, clf, test_acc_data)
    
    print ("Training accuracies")
    train_acc_data = [training_features, training_labels, training_features, training_labels, costs]
    fitness(fittest, clf, train_acc_data)
    
if __name__ == "__main__":
    main()