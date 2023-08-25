import customtkinter

def button_callback():
    print("button pressed")

app = customtkinter.CTk()
app.title("my app")
# app.geometry("400x150")
app.grid_columnconfigure((0, 1), weight=1)

left_frame = customtkinter.CTkFrame(app)
left_frame.grid_columnconfigure((0, 1), weight=1)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')

button = customtkinter.CTkButton(left_frame, text="Open Data", command=button_callback)
button.grid(row=0, column=0, padx=10, pady=10, sticky='ew', columnspan=2)

header_label = customtkinter.CTkLabel(left_frame, text='Header Lines', anchor="w")
header_label.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

header_entry = customtkinter.CTkEntry(left_frame)
header_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

verbosity_checkbox = customtkinter.CTkCheckBox(left_frame, text="Increase Verbosity")
verbosity_checkbox.grid(row=2, column=0, padx=10, pady=10, sticky='ew', columnspan=2)

dim_frame = customtkinter.CTkFrame(left_frame)
dim_frame.grid(row=3, column=0, padx=10, pady=10, sticky='ew', columnspan=2)
dim_frame.grid_columnconfigure((0, 1, 2), weight=1)

dim_label = customtkinter.CTkLabel(dim_frame, text='Dimensions', anchor="w")
dim_label.grid(row=0, column=0, padx=10, pady=10, sticky='ew', columnspan=3)

x_entry = customtkinter.CTkEntry(dim_frame, placeholder_text='X', width=70)
x_entry.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
y_entry = customtkinter.CTkEntry(dim_frame, placeholder_text='Y', width=70)
y_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
z_entry = customtkinter.CTkEntry(dim_frame, placeholder_text='Z', width=70)
z_entry.grid(row=1, column=2, padx=10, pady=10, sticky='ew')

app.mainloop()