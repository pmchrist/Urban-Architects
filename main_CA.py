import argparse
from CA_Model import function_one, function_two

def main():
    parser = argparse.ArgumentParser(description="Example program")
    parser.add_argument('--function', choices=['simple', 'complicated'], help='Specify function to execute')

    args = parser.parse_args()

    if args.function == 'simple':
        function_one()
        print("Simulation complete!")
    elif args.function == 'complicated':
        function_two()
        print("Simulation complete!")   
    else:
        print("Invalid function choice!")

if __name__ == '__main__':
    main()
