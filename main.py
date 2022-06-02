from lib.drill_core_process import DrillCoreProcess


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    dcp = DrillCoreProcess()
    dcp.process_image(image_name = 'img.jpeg')
    # dcp.process_image(image_name = 'img2.jpeg')
    # dcp.process_image(image_name = 'img3.jpeg')

    # dcp.split_drill_core()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
