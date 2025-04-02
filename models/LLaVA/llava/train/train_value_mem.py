from llava.train.train_value import train_value

if __name__ == "__main__":
    train_value(attn_implementation="flash_attention_2")