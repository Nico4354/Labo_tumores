import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import axios from 'axios';

// Asegúrate de que esta sea exactamente la URL que te dio Render para tu proyecto
const API_URL = 'https://labo-tumores.onrender.com';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  image: File | null = null;
  prediction: string = '';
  imageName: string = '';
  probs: { [key: string]: number } | null = null;
  imageUrl: string = '';
  graphUrl: string = '';

  handleFileChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.image = input.files[0];
      this.prediction = '';
      this.imageName = '';
      this.probs = null;
      this.imageUrl = '';
      this.graphUrl = '';
    }
  }

  async handleSubmit() {
    if (!this.image) return;

    const formData = new FormData();
    formData.append('image', this.image);

    try {
      const res = await axios.post(`${API_URL}/api/clasificar`, formData);
      const data = res.data;
      
      this.prediction = data.prediction;
      this.imageName = data.image_name;
      this.probs = data.probs;
      
      // ACTUALIZADO: Usamos las rutas directas '/uploads/' que configuramos en Python
      this.imageUrl = `${API_URL}/uploads/${this.imageName}`;
      
      // ACTUALIZADO: Leemos el nombre exacto e irrepetible del gráfico que nos manda el backend
      this.graphUrl = `${API_URL}/uploads/${data.graph_name}`; 
      
    } catch (err) {
      alert('Error al clasificar la imagen. Revisa la consola para más detalles.');
      console.error(err);
    }
  }

  getProbs() {
    return this.probs ? Object.entries(this.probs) : [];
  }
}