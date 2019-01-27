import argparse

from opengnn import constants
from opengnn.config import load_subtokenizer
from opengnn.utils.vocab import Vocab

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data", nargs="+",
        help="Source text file.")
    parser.add_argument(
        "--save_vocab", required=True,
        help="Output vocabulary file.")
    parser.add_argument(
        "--field_name", default=None,
        help="The field name in the json file that contains the tokens.")
    parser.add_argument(
        "--string_index", type=int, default=None,
        help="If a token is represented as tuple of information, this index represents"
        "the location in the tuple we want to vectorize")
    parser.add_argument(
        "--subtokenizer", default=None,
        help="A python file with a `subtokenize` function."
             "Vocabulary will be build based on subtokens produced by this function")
    parser.add_argument(
        "--min_frequency", type=int, default=1,
        help="Minimum word frequency.")
    parser.add_argument(
        "--size", type=int, default=0,
        help="Maximum vocabulary size. If = 0, do not limit vocabulary.")
    parser.add_argument(
        "--with_sequence_tokens", default=False, action="store_true",
        help="If set, add special sequence tokens (start, end)"
        "in the vocabulary.")
    parser.add_argument(
        "--no_pad_token", default=False, action="store_true",
        help="If set, do not add special pad token"
        "in the vocabulary.")
    parser.add_argument(
        "--case_sensitive", default=False, action="store_true",
        help="If set, vocabulary building will take in consideration casing of words")
    args = parser.parse_args()

    special_tokens = []
    if not args.no_pad_token:
        special_tokens.append(constants.PADDING_TOKEN)
    if args.with_sequence_tokens:
        special_tokens.append(constants.START_OF_SENTENCE_TOKEN)
        special_tokens.append(constants.END_OF_SENTENCE_TOKEN)

    vocab = Vocab(special_tokens=special_tokens)
    if args.subtokenizer is not None:
        subtokenizer = load_subtokenizer(args.subtokenizer)
    else:
        subtokenizer = None

    for data_file in args.data:
        vocab.add_from_file(
            data_file, args.field_name, args.string_index, subtokenizer, args.case_sensitive)
    vocab = vocab.prune(max_size=args.size, min_frequency=args.min_frequency)
    vocab.serialize(args.save_vocab)


if __name__ == "__main__":
    main()
