import subprocess
import tkinter as tk

def ping_host():
    command = entry.get()
    
    # Create a subprocess and start the ping command
    ping_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    previous_line = ''

    # Read the output dynamically and update the text in the Tkinter window
    for line in iter(ping_process.stdout.readline, b''):
        line = line.decode('utf-8').strip()
        if line.startswith("⚊") or "⚊" in line:
            text.delete('end - %d chars' % (len(previous_line) + 2), 'end-1c')
        text.insert(tk.END, line + '\n')
        text.see(tk.END)
        root.update()
        previous_line = line

# Create the Tkinter window
root = tk.Tk()
root.title("Ping Command Output")

root.geometry('{}x{}'.format(1000, 1000))

# Create an entry field for the hostname
entry = tk.Entry(root)
entry.pack()

# Create a text widget to display the ping output
text = tk.Text(root, height=50, width=100)
text.pack()

# Create a button to start the ping command
button = tk.Button(root, text="Ping", command=ping_host)
button.pack()

root.mainloop()