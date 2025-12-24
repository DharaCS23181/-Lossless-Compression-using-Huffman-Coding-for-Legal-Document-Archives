import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import heapq
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

ARCHIVE_FOLDER = "Huffman_Archives"
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

# ---------------- Node for Huffman Tree ----------------
class Node:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq


# ---------------- Huffman Code Building ----------------
def build_huffman_codes(freq_dict):
    heap = []
    for sym, fr in freq_dict.items():
        heapq.heappush(heap, (fr, Node(sym, fr)))
    if len(heap) == 1:
        fr, node = heapq.heappop(heap)
        root = Node(freq=fr)
        root.left = node
        return {node.symbol: '0'}, root
    while len(heap) > 1:
        fr1, n1 = heapq.heappop(heap)
        fr2, n2 = heapq.heappop(heap)
        merged = Node(freq=fr1 + fr2)
        merged.left, merged.right = n1, n2
        heapq.heappush(heap, (merged.freq, merged))
    root = heapq.heappop(heap)[1]
    codes = {}
    def assign_codes(node, code_str):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = code_str or '0'
            return
        assign_codes(node.left, code_str + '0')
        assign_codes(node.right, code_str + '1')
    assign_codes(root, "")
    return codes, root


# ---------------- Huffman Compression ----------------
def huffman_compress_bytes(data_bytes):
    freq = {}
    for b in data_bytes:
        freq[b] = freq.get(b, 0) + 1
    codes, _ = build_huffman_codes(freq)
    encoded_bits = ''.join(codes[b] for b in data_bytes)
    byte_arr = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i + 8]
        if len(byte) < 8:
            byte = byte.ljust(8, '0')
        byte_arr.append(int(byte, 2))
    metadata = {'codes': codes, 'bits_len': len(encoded_bits)}
    return byte_arr, metadata


# ---------------- Huffman Decompression ----------------
def huffman_decompress_bytes(byte_arr, metadata):
    encoded_bits = ''.join(format(b, '08b') for b in byte_arr)
    encoded_bits = encoded_bits[:metadata['bits_len']]
    rev_codes = {v: k for k, v in metadata['codes'].items()}
    decoded_bytes = bytearray()
    temp = ''
    for bit in encoded_bits:
        temp += bit
        if temp in rev_codes:
            decoded_bytes.append(rev_codes[temp])
            temp = ''
    return decoded_bytes


# ---------------- Helpers for Histogram, Entropy, Variance ----------------
def compute_byte_histogram_from_bytes(data_bytes):
    if len(data_bytes) == 0:
        return np.zeros(256, dtype=int)
    hist, _ = np.histogram(list(data_bytes), bins=range(257))
    return hist

def compute_byte_histogram(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return compute_byte_histogram_from_bytes(data)

def compute_entropy_from_bytes(data_bytes):
    if len(data_bytes) == 0:
        return 0.0
    freq = np.bincount(list(data_bytes), minlength=256)
    probs = freq / freq.sum()
    probs_nonzero = probs[probs > 0]
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    return float(entropy)

def compute_variance_from_bytes(data_bytes):
    hist = compute_byte_histogram_from_bytes(data_bytes)
    return float(np.var(hist))


# ---------------- Main GUI ----------------
class HuffmanPDFGUI:
    def __init__(self, master):
        self.master = master
        master.title("Huffman PDF Compressor (Enhanced Metrics)")
        master.geometry("950x780")
        master.configure(bg="#f0f0f0")

        self.pdf_path = None
        self.canvas = None
        self.orig_pdf_size = 0

        tk.Label(master, text="📘 Huffman PDF Compressor", font=("Arial", 18, "bold"), bg="#f0f0f0").pack(pady=10)
        tk.Button(master, text="📂 Select PDF", command=self.select_pdf, bg="#4caf50", fg="white", padx=10, pady=6).pack(pady=5)
        tk.Button(master, text="🗜️ Compress PDF", command=self.compress_action, bg="#2196f3", fg="white", padx=10, pady=6).pack(pady=5)
        tk.Button(master, text="📄 Restore PDF", command=self.restore_action, bg="#ff9800", fg="white", padx=10, pady=6).pack(pady=5)

        self.lbl_info = tk.Label(master, text="No file selected", bg="#f0f0f0", font=("Arial", 12))
        self.lbl_info.pack(pady=10)
        self.lbl_stats = tk.Label(master, text="", bg="#f0f0f0", font=("Arial", 12), justify="center")
        self.lbl_stats.pack(pady=10)

    # ---------------- Select PDF ----------------
    def select_pdf(self):
        path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path = path
            self.orig_pdf_size = os.path.getsize(path)
            self.lbl_info.config(text=f"Selected: {os.path.basename(path)} | Size: {self.orig_pdf_size / 1024:.2f} KB")

    # ---------------- Compress Action ----------------
    def compress_action(self):
        if not self.pdf_path:
            messagebox.showwarning("Warning", "Please select a PDF first.")
            return
        save_huff = os.path.join(ARCHIVE_FOLDER, os.path.basename(self.pdf_path).replace(".pdf", ".bin"))
        self.lbl_stats.config(text="Compressing... Please wait.")
        self.master.update()

        try:
            start = time.time()
            with open(self.pdf_path, "rb") as f:
                pdf_bytes = f.read()
            compressed_bytes, metadata = huffman_compress_bytes(pdf_bytes)
            with open(save_huff, "wb") as f:
                pickle.dump({"bytes": compressed_bytes, "metadata": metadata}, f)
            huff_size = os.path.getsize(save_huff)
            end = time.time()

            ratio = huff_size / self.orig_pdf_size
            entropy = compute_entropy_from_bytes(pdf_bytes)
            variance = compute_variance_from_bytes(pdf_bytes)
            compression_percent = (1 - ratio) * 100

            self.lbl_stats.config(
                text=(
                    f"Original Size: {self.orig_pdf_size / 1024:.2f} KB\n"
                    f"Compressed Size: {huff_size / 1024:.2f} KB\n"
                    f"Compression Ratio: {ratio:.3f} | {compression_percent:.2f}% smaller\n\n"
                    f"Entropy: {entropy:.4f} bits/symbol\n"
                    f"Byte Frequency Variance: {variance:.2f}\n"
                    f"Time Taken: {end - start:.2f} sec"
                )
            )
            messagebox.showinfo("Success", f"PDF compressed to:\n{save_huff}")
            self.show_graph(self.orig_pdf_size, huff_size)
        except Exception as e:
            messagebox.showerror("Error", f"Compression failed: {e}")

    # ---------------- Restore Action ----------------
    def restore_action(self):
        files = [f for f in os.listdir(ARCHIVE_FOLDER) if f.endswith(".bin")]
        if not files:
            messagebox.showwarning("Warning", "No Huffman archive files found.")
            return

        huff_file = filedialog.askopenfilename(initialdir=ARCHIVE_FOLDER, title="Select Huffman File", filetypes=[("Huffman files", "*.bin")])
        if not huff_file:
            return

        save_pdf = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")], title="Save Restored PDF As")
        if not save_pdf:
            return

        self.lbl_stats.config(text="Restoring PDF... Please wait.")
        self.master.update()

        try:
            start = time.time()
            with open(huff_file, "rb") as f:
                data = pickle.load(f)
            restored_bytes = huffman_decompress_bytes(data["bytes"], data["metadata"])
            with open(save_pdf, "wb") as f:
                f.write(restored_bytes)
            restored_size = os.path.getsize(save_pdf)
            end = time.time()

            entropy = compute_entropy_from_bytes(restored_bytes)
            variance = compute_variance_from_bytes(restored_bytes)

            self.lbl_stats.config(
                text=(
                    f"Restored PDF Size: {restored_size / 1024:.2f} KB\n"
                    f"Entropy: {entropy:.4f} bits/symbol\n"
                    f"Byte Frequency Variance: {variance:.2f}\n"
                    f"Restoration Time: {end - start:.2f} sec"
                )
            )
            messagebox.showinfo("Success", f"Restored PDF saved:\n{save_pdf}")
            self.show_graph(self.orig_pdf_size, restored_size)
        except Exception as e:
            messagebox.showerror("Error", f"Restoration failed: {e}")

    # ---------------- Graph Display ----------------
    def show_graph(self, size1, size2):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Bar Chart
        axes[0].bar(['Original', 'Compared'], [size1 / 1024, size2 / 1024], color=['blue', 'green'])
        axes[0].set_ylabel('Size (KB)')
        axes[0].set_title('PDF Size Comparison')
        for i, v in enumerate([size1 / 1024, size2 / 1024]):
            axes[0].text(i, v + 5, f"{v:.2f} KB", ha='center')

        # Histogram
        try:
            orig_hist = compute_byte_histogram(self.pdf_path)
            comp_file = os.path.join(ARCHIVE_FOLDER, os.path.basename(self.pdf_path).replace(".pdf", ".bin"))
            if os.path.exists(comp_file):
                with open(comp_file, "rb") as f:
                    comp_bytes = pickle.load(f)["bytes"]
                comp_hist, _ = np.histogram(list(comp_bytes), bins=range(257))
            else:
                comp_hist = np.zeros(256)

            axes[1].plot(orig_hist, color='blue', label='Original PDF')
            axes[1].plot(comp_hist, color='green', label='Compressed BIN')
            axes[1].set_title('Byte Frequency Histogram')
            axes[1].set_xlabel('Byte Value (0–255)')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Histogram Error:\n{e}", ha='center', va='center', color='red')

        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)


# ---------------- Run GUI ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HuffmanPDFGUI(root)
    root.mainloop()
