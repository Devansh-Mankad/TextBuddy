import json
from pipeline import process

def main():
    text=input("\nEnter a paragraph:\n> ").strip()
    if not text:
        text="The student recieve the file. He sent it because the teacher asked."
    result=process(text)
    print("\nCorrected text:")
    print(result["corrected_text"])
    print("\nSummary:")
    print(json.dumps(result["summary"],indent=2))
    print("\nFull result:")
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":
    main()
