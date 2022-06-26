import pytest


def fourOne(c):
    if c >= 4 and c <= 15:
        print("finish setting")


def practice_2(g):
    # Requriements
    # The input Value f(g) return from 4.35 to 8.2 as normal vaules.
    # The input value out of the range above is recognized as abnormal values

    # Question1
    # What would you ask an enginner who has written the above the requirement.
    # My answer
    # Check if the significant value is 8.20
    # Check if 4.35 and 8.20 are included in the normal values
    # Question2
    # List up seven kinds of values when you test the above requirements
    # in the following two ways.
    # Method13
    # Solve 4 kinds of test values to execute Equivalence partition and boarder analysis.
    # Considering hidden boarder values, solve 3 kinds of test value,
    # where the history of document does not exist and you can not ask the engineer about the history.
    # My answer2
    # I have no idea, bitch.( The answer was 5.0, 6.0 and 7.0! Who the fuck can guess the answer!!!!!)

    if g >= 4.35 and g <= 8.20:
        print("Normal Value")


def test_poe():
    print("a")
    a = 3
    assert a == 3


def practice_3(age):
    print("Let's start pracetice3!")
    print("The entrance fee of WaiWai Aquarium is shown in the following table.")
    print("Find boarder values after dividing it into Equivalence partition")
    # Table:: The entrance fee of WaiWAi Aquarium.
    # Adult(grater than or equal to 16 years old):2000 yen
    # Junior(grater than or equal to 7years old): 900 yen
    # Child(grater than or equal to 4years old): 400yen
    # Infant(less than 4 years old): 0yen
    if age >= 16:
        print("print")
    elif age >= 7 and age < 16:
        print("900")
    elif age >= 4 and age < 7:
        print("400")
    elif age < 4:
        print("0")


def price_rental(price, age, new):

    if new == False:
        print("50 percent off")
        ans = price * 0.5  # if a DVD is old
    else:
        # Age
        if age <= 18:  # if the age is greater than or equal to 18
            print("10 percent off")
            ans = price * 0.90
        elif age >= 65:
            print("20 percent off")
            ans = price * 0.80
        else:
            ans = price
    return ans


@pytest.mark.parametrize(
    ("price", "age", "new", "expected"),
    [
        (1000, 17, False, 500),
        (500, 18, True, 450),
        (2000, 65, False, 1000),
        (1200, 66, True, 960),
        (2000,19,False,1000),
        (1000,64,True,1000)
    ],
)
def test_price_rental(price, age, new, expected):
    assert price_rental(price, age, new) == expected
