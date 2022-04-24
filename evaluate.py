import architecture
from digit import *
from utils.trainutil import test_highdim_dist, test_category_no_loss

if __name__ == "__main__":
    # Params setup
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--model_weight", type=str, required=True, help='path to trained model weight')
    parser.add_argument("--model", type=str, required=True, help='model type')
    parser.add_argument("--label", type=str, required=True, help='label type')
    parser.add_argument(
        "--label_dir",
        type=str,
        help="Directory where labels are stored",
        default=None,
    )
    parser.add_argument("--dataset", type=str, help="Dataset to test on")

    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size
    label_dir = args.label_dir
    dataset = args.dataset
    label = args.label
    num_workers = 4
    num_classes = 10
    assert(dataset in ("m", "s", "u"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if "category" in label or label in ("lowdim", "glove"):
        if label == "glove":
            model = architecture.CategoryModel(model_name, 50)
        else:
            model = architecture.CategoryModel(model_name, num_classes)
    elif label == "bert":
        model = architecture.BERTHighDimensionalModel(model_name, num_classes)
    else:
        model = architecture.HighDimensionalModel(model_name, num_classes)

    model = model.to(device)

    testloaders = digit_load_test(batch_size, dataset, label_dir)

    if "category" in label:
        acc = test_category_no_loss(model, testloaders, device)
    else:
        _, acc = test_highdim_dist(model, testloaders, device)

    print('test accuracy: ', acc)