import numpy as np
def main():
    datasets = ['AbnormalHeartbeat', "SelfRegulationSCP2", "AppliancesEnergy", "LiveFuelMoistureContent"]
    for data in datasets:
        train = np.load(f"C:/Users/AlphaNHH/Downloads/Time Series Data/{data}/X_train_original.npy")
        test = np.load(f"C:/Users/AlphaNHH/Downloads/Time Series Data/{data}/X_test_original.npy")
        print('-'*25+data+'-'*25)
        print(f"Train Shape: {train.shape}")
        print(f"Test Shape: {test.shape}")
        print("-"*50)


if __name__ == "__main__":
    main()
