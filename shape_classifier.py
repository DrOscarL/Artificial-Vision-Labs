#%% EJERCICIO 1: Clasificador de formas (círculo, cuadrado, triángulo) en video
import numpy as np
import cv2
from collections import deque

class ShapeClassifier:
    def __init__(self, use_realsense=True):
        self.use_realsense = use_realsense
        
        # Estadísticas
        self.shape_counts = {'circulo': 0, 'cuadrado': 0, 'triangulo': 0, 
                             'rectangulo': 0, 'otro': 0}
        self.shape_history = deque(maxlen=100)
        
        if use_realsense:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(self.config)
            
            for _ in range(30):
                self.pipeline.wait_for_frames()
        else:
            self.cap = cv2.VideoCapture(0)
    
    def get_frame(self):
        """Obtiene frame según cámara"""
        if self.use_realsense:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image, depth_frame
        else:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            return frame, None, None
    
    def preprocess_frame(self, color_image, depth_image=None):
        """
        PASO 1: PREPROCESAMIENTO
        Aplica filtros y segmentación
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # FILTRADO: Reducir ruido con Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Si hay profundidad, usarla para segmentación
        if depth_image is not None:
            # Segmentar por profundidad (objetos cercanos)
            mask_depth = np.logical_and(
                depth_image > 400, 
                depth_image < 1200
            ).astype(np.uint8) * 255
            
            # Combinar con threshold de intensidad
            _, mask_intensity = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            
            # Combinar máscaras
            mask = cv2.bitwise_and(mask_intensity, mask_depth)
        else:
            # Solo usar threshold de intensidad
            _, mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        # MORFOLOGÍA: Limpiar máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return blurred, mask
    
    def detect_edges(self, blurred):
        """
        PASO 2: DETECCIÓN DE BORDES
        Aplica Canny para encontrar contornos
        """
        edges = cv2.Canny(blurred, 50, 150)
        return edges
    
    def classify_shape(self, contour):
        """
        PASO 3: CLASIFICACIÓN DE FORMA
        Usa análisis de contorno y aproximación poligonal
        """
        # Calcular perímetro
        perimeter = cv2.arcLength(contour, True)
        
        # Aproximación poligonal (Douglas-Peucker)
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Número de vértices
        vertices = len(approx)
        
        # Calcular área y circularidad
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # CLASIFICACIÓN
        shape_name = "otro"
        
        if vertices == 3:
            shape_name = "triangulo"
        
        elif vertices == 4:
            # Distinguir cuadrado de rectángulo
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "cuadrado"
            else:
                shape_name = "rectangulo"
        
        elif vertices > 8:
            # Muchos vértices → probablemente círculo
            if circularity > 0.7:
                shape_name = "circulo"
        
        # Información adicional
        info = {
            'name': shape_name,
            'vertices': vertices,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'approx': approx
        }
        
        return info
    
    def calculate_dimensions(self, contour, depth_frame=None):
        """
        PASO 4: MEDICIÓN (si hay profundidad)
        Calcula dimensiones reales usando profundidad
        """
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w//2, y + h//2
        
        dimensions = {
            'width_px': w,
            'height_px': h,
            'center': (cx, cy)
        }
        
        if depth_frame is not None:
            # Obtener distancia
            distance = depth_frame.get_distance(cx, cy)
            
            if distance > 0:
                # Obtener intrínsecos (necesario para conversión px→mm)
                intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                
                # Factor de conversión píxeles a mm a esta distancia
                # Aproximación: 1 píxel = distance/fx (en metros)
                pixels_to_meters = distance / intrinsics.fx
                
                dimensions['width_m'] = w * pixels_to_meters
                dimensions['height_m'] = h * pixels_to_meters
                dimensions['distance_m'] = distance
        
        return dimensions
    
    def run(self):
        """Pipeline principal en tiempo real"""
        print("="*50)
        print("CLASIFICADOR DE FORMAS GEOMÉTRICAS")
        print("="*50)
        print("\nInstrucciones:")
        print("  • Coloca objetos con formas geométricas claras")
        print("  • El sistema clasificará: círculo, cuadrado, triángulo, rectángulo")
        print("  • Presiona 'r' para resetear contadores")
        print("  • Presiona 's' para guardar frame")
        print("  • Presiona 'q' para salir\n")
        
        frame_count = 0
        
        # Crear ventana con trackbars para ajustar parámetros
        cv2.namedWindow('Clasificador')
        cv2.createTrackbar('Min Area', 'Clasificador', 500, 5000, lambda x: None)
        cv2.createTrackbar('Threshold', 'Clasificador', 60, 255, lambda x: None)
        
        while True:
            # 1. CAPTURA
            color_image, depth_image, depth_frame = self.get_frame()
            
            # Obtener parámetros ajustables
            min_area = cv2.getTrackbarPos('Min Area', 'Clasificador')
            threshold_val = cv2.getTrackbarPos('Threshold', 'Clasificador')
            
            # 2. PREPROCESAMIENTO
            blurred, mask = self.preprocess_frame(color_image, depth_image)
            
            # 3. DETECCIÓN DE BORDES (para visualización)
            edges = self.detect_edges(blurred)
            
            # 4. ENCONTRAR CONTORNOS
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # 5. ANÁLISIS Y CLASIFICACIÓN
            result = color_image.copy()
            shapes_in_frame = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por área mínima
                if area > min_area:
                    # CLASIFICAR
                    shape_info = self.classify_shape(contour)
                    
                    # MEDIR (si hay profundidad)
                    dimensions = self.calculate_dimensions(contour, depth_frame)
                    
                    # Guardar en lista
                    shapes_in_frame.append(shape_info['name'])
                    
                    # VISUALIZACIÓN
                    # Dibujar contorno
                    color = self.get_shape_color(shape_info['name'])
                    cv2.drawContours(result, [contour], -1, color, 2)
                    
                    # Dibujar aproximación poligonal
                    cv2.drawContours(result, [shape_info['approx']], -1, (0, 255, 255), 2)
                    
                    # Bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x+w, y+h), color, 1)
                    
                    # Información de la forma
                    cx, cy = dimensions['center']
                    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Etiqueta con información
                    label = f"{shape_info['name']}"
                    if 'distance_m' in dimensions:
                        label += f"\n{dimensions['distance_m']:.2f}m"
                        label += f"\n{dimensions['width_m']*100:.1f}cm"
                    
                    # Dibujar texto multilínea
                    y_offset = y - 10
                    for i, line in enumerate(label.split('\n')):
                        cv2.putText(result, line, (x, y_offset - i*20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Info adicional: vértices y circularidad
                    info_text = f"V:{shape_info['vertices']} C:{shape_info['circularity']:.2f}"
                    cv2.putText(result, info_text, (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            # Actualizar contadores
            for shape in shapes_in_frame:
                if shape in self.shape_counts:
                    self.shape_history.append(shape)
            
            # PANEL DE INFORMACIÓN
            self.draw_info_panel(result, shapes_in_frame)
            
            # VISUALIZACIONES ADICIONALES
            # Crear panel con vistas múltiples
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Resize para panel
            h, w = color_image.shape[:2]
            small_h, small_w = h//3, w//3
            
            mask_small = cv2.resize(mask_colored, (small_w, small_h))
            edges_small = cv2.resize(edges_colored, (small_w, small_h))
            
            # Colocar mini vistas en esquina
            result[0:small_h, 0:small_w] = mask_small
            result[0:small_h, small_w:small_w*2] = edges_small
            
            # Etiquetas de mini vistas
            cv2.putText(result, "Mask", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(result, "Edges", (small_w+5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            cv2.imshow('Clasificador', result)
            
            frame_count += 1
            
            # CONTROLES
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.shape_counts = {k: 0 for k in self.shape_counts}
                self.shape_history.clear()
                print("✓ Contadores reseteados")
            elif key == ord('s'):
                filename = f'shapes_frame_{frame_count}.png'
                cv2.imwrite(filename, result)
                print(f"✓ Guardado: {filename}")
        
        # Estadísticas finales
        self.print_statistics()
        
        self.cleanup()
        cv2.destroyAllWindows()
    
    def get_shape_color(self, shape_name):
        """Colores por forma"""
        colors = {
            'circulo': (255, 0, 0),      # Azul
            'cuadrado': (0, 255, 0),     # Verde
            'triangulo': (0, 0, 255),    # Rojo
            'rectangulo': (255, 255, 0), # Cian
            'otro': (128, 128, 128)      # Gris
        }
        return colors.get(shape_name, (255, 255, 255))
    
    def draw_info_panel(self, image, current_shapes):
        """Dibuja panel de información"""
        h, w = image.shape[:2]
        
        # Panel semi-transparente
        overlay = image.copy()
        panel_h = 150
        cv2.rectangle(overlay, (0, h-panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Título
        cv2.putText(image, "FORMAS DETECTADAS", (10, h-panel_h+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Contadores
        x_offset = 10
        y_base = h - panel_h + 60
        
        for shape_name, color in [('circulo', (255, 0, 0)), 
                                   ('cuadrado', (0, 255, 0)),
                                   ('triangulo', (0, 0, 255)), 
                                   ('rectangulo', (255, 255, 0))]:
            
            count = current_shapes.count(shape_name)
            
            # Símbolo
            if shape_name == 'circulo':
                cv2.circle(image, (x_offset+15, y_base), 12, color, 2)
            elif shape_name == 'cuadrado':
                cv2.rectangle(image, (x_offset+3, y_base-12), 
                            (x_offset+27, y_base+12), color, 2)
            elif shape_name == 'triangulo':
                pts = np.array([[x_offset+15, y_base-12], 
                               [x_offset+3, y_base+12], 
                               [x_offset+27, y_base+12]], np.int32)
                cv2.polylines(image, [pts], True, color, 2)
            elif shape_name == 'rectangulo':
                cv2.rectangle(image, (x_offset+3, y_base-8), 
                            (x_offset+27, y_base+8), color, 2)
            
            # Texto
            cv2.putText(image, f"{shape_name}: {count}", (x_offset+35, y_base+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            x_offset += 150
        
        # Total
        cv2.putText(image, f"Total: {len(current_shapes)}", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Historial (últimas 100 detecciones)
        if len(self.shape_history) > 0:
            history_text = f"Historial: "
            for shape in ['circulo', 'cuadrado', 'triangulo', 'rectangulo']:
                count = list(self.shape_history).count(shape)
                history_text += f"{shape[0].upper()}:{count} "
            
            cv2.putText(image, history_text, (w-400, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def print_statistics(self):
        """Imprime estadísticas finales"""
        print("\n" + "="*50)
        print("ESTADÍSTICAS FINALES")
        print("="*50)
        
        total = len(self.shape_history)
        
        if total > 0:
            for shape in ['circulo', 'cuadrado', 'triangulo', 'rectangulo']:
                count = list(self.shape_history).count(shape)
                percentage = (count / total) * 100
                print(f"{shape.capitalize()}: {count} ({percentage:.1f}%)")
            
            print(f"\nTotal de detecciones: {total}")
        else:
            print("No se detectaron formas")
    
    def cleanup(self):
        if self.use_realsense:
            self.pipeline.stop()
        else:
            self.cap.release()

# EJECUTAR
classifier = ShapeClassifier(use_realsense=True)
classifier.run()

# O con webcam:
# classifier = ShapeClassifier(use_realsense=False)
# classifier.run()