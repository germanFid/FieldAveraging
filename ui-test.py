import customtkinter

JOB_CONFIG = [["Basic 2D","basic_2d"], ["Basic 2D Parallel", "basic2d_paral"], 
              ["Basic 3D", "basic3d"], ["Basic 3D Parallel","basic3d_paral"], ["Gauss", "gauss"]]

def button_callback():
    print("button pressed")

app = customtkinter.CTk()
app.title("Averager GUI - Blooming Lycoris")
app.resizable(False, False)
# app.geometry("400x150")
app.grid_columnconfigure((0, 1), weight=1)

left_frame = customtkinter.CTkFrame(app, fg_color="transparent")
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
dim_label.grid(row=0, column=0, padx=10, pady=5, sticky='ew', columnspan=3)

x_entry = customtkinter.CTkEntry(dim_frame, placeholder_text='X', width=70)
x_entry.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
y_entry = customtkinter.CTkEntry(dim_frame, placeholder_text='Y', width=70)
y_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
z_entry = customtkinter.CTkEntry(dim_frame, placeholder_text='Z', width=70)
z_entry.grid(row=1, column=2, padx=10, pady=10, sticky='ew')

job_frame = customtkinter.CTkFrame(left_frame)
job_frame.grid(row=4, column=0, padx=10, pady=10, sticky='ew', columnspan=2)

job_enrties = []

job_label = customtkinter.CTkLabel(job_frame, text='Job', anchor="w")
job_label.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

job_tabview = customtkinter.CTkTabview(job_frame)
job_tabview.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

cpu_tab = job_tabview.add("CPU")
gpu_tab = job_tabview.add("GPU")
job_tabview.set("CPU")

for i, entry in enumerate(JOB_CONFIG):
    job_checkbox = customtkinter.CTkCheckBox(cpu_tab, text=entry[0])
    job_checkbox.grid(row=i, column=0, padx=10, pady=10, sticky='ew')
    job_enrties.append(job_checkbox)

radius_label = customtkinter.CTkLabel(left_frame, text='Radius', anchor="w")
radius_label.grid(row=5, column=0, padx=10, pady=10, sticky='ew')

radius_entry = customtkinter.CTkEntry(left_frame)
radius_entry.grid(row=5, column=1, padx=10, pady=10, sticky='ew')

iter_label = customtkinter.CTkLabel(left_frame, text='Iterations', anchor="w")
iter_label.grid(row=6, column=0, padx=10, pady=10, sticky='ew')

iter_entry = customtkinter.CTkEntry(left_frame)
iter_entry.grid(row=6, column=1, padx=10, pady=10, sticky='ew')

col_label = customtkinter.CTkLabel(left_frame, text='Key Columns', anchor="w")
col_label.grid(row=7, column=0, padx=10, pady=10, sticky='ew')

col_entry = customtkinter.CTkEntry(left_frame)
col_entry.grid(row=7, column=1, padx=10, pady=10, sticky='ew')

run_button = customtkinter.CTkButton(left_frame, text="Run!", command=button_callback)
run_button.grid(row=8, column=0, padx=10, pady=10, sticky='ew', columnspan=2)

app.mainloop()