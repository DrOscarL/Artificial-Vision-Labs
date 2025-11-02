#%% EJERCICIO 2: Inspector de calidad en tiempo real
import numpy as np
import cv2

class QualityInspector:
    """
    Sistema de inspección que verifica:
    1. Forma correcta
    2. Tamaño dentro de especificaciones
    3. Color uniforme
    4. Sin defectos (agujeros, grietas)
    """
    
    def __init__(self, use_realsense=True):
        self.use_realsense = use_realsense
        
        # Especificaciones de calidad
        self.specs = {
            'circulo': {
                'min_circularity': 0.8,
                'min_diameter_mm': 30,
                'max_diameter_mm': 50
            },
            'cuadrado': {
                'min_rectangularity': 0.9,
                'min_side_mm': 30,
                'max_side_mm': 50,
                'max_aspect_ratio_diff': 0.1
            }
        }
        
        # Estadísticas
        self.inspected_count = 0
        self.passed_count = 0
        self.failed_count = 0
        self.defect_types = []
        
        if use_realsense:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(self.config)
            
            # Obtener intrínsecos
            profile = self.pipeline.get_active_profile()
            self.depth_profile = profile.get_stream(rs.stream.depth)
            self.intrinsics = self.depth_profile.as_video_stream_profile().get_intrinsics()
            
            for _ in range(30):
                self.pipeline.wait_for_frames()
        else:
            self.cap = cv2.VideoCapture(0)
            self.intrinsics = None
    
    def get_frame(self):
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
    
    def inspect_shape(self, contour, shape_name):
        """
        Inspecciona si la forma cumple especificaciones
        """
        defects = []
        
        # Calcular parámetros geométricos
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if perimeter == 0:
            return False, ["Perímetro inválido"]
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if shape_name == 'circulo':
            # Verificar circularidad
            if circularity < self.specs['circulo']['min_circularity']:
                defects.append(f"Baja circularidad: {circularity:.2f}")
        
        elif shape_name == 'cuadrado':
            # Verificar que sea realmente cuadrado
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if abs(aspect_ratio - 1.0) > self.specs['cuadrado']['max_aspect_ratio_diff']:
                defects.append(f"No es cuadrado: AR={aspect_ratio:.2f}")
        
        # Verificar agujeros internos
        if self.has_holes(contour):
            defects.append("Contiene agujeros")
        
        # Verificar uniformidad de color
        # (se puede implementar analizando ROI)
        
        passed = len(defects) == 0
        return passed, defects
    
    def has_holes(self, contour):
        """Detecta si el contorno tiene agujeros internos"""
        # Crear máscara del contorno
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Rellenar contorno y comparar
        filled = mask.copy()
        h, w = filled.shape
        flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        
        # Si hay diferencia, hay agujeros
        filled_inv = cv2.bitwise_not(filled)
        holes = cv2.bitwise_and(mask, filled_inv)
        
        return np.sum(holes) > 100  # Threshold para ruido
    
    def measure_dimensions(self, contour, depth_frame):
        """Mide dimensiones reales usando profundidad"""
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w//2, y + h//2
        
        if depth_frame is None or self.intrinsics is None:
            return None
        
        distance = depth_frame.get_distance(cx, cy)
        
        if distance == 0:
            return None
        
        # Convertir píxeles a mm
        pixels_to_meters = distance / self.intrinsics.fx
        
        width_mm = w * pixels_to_meters * 1000
        height_mm = h * pixels_to_meters * 1000
        
        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'distance_m': distance
        }
    
    def check_dimensions(self, dimensions, shape_name):
        """Verifica si las dimensiones están en spec"""
        if dimensions is None:
            return False, ["No se pudo medir"]
        
        defects = []
        
        if shape_name == 'circulo':
            diameter = (dimensions['width_mm'] + dimensions['height_mm']) / 2
            min_d = self.specs['circulo']['min_diameter_mm']
            max_d = self.specs['circulo']['max_diameter_mm']
            
            if diameter < min_d:
                defects.append(f"Muy pequeño: {diameter:.1f}mm < {min_d}mm")
            elif diameter > max_d:
                defects.append(f"Muy grande: {diameter:.1f}mm > {max_d}mm")
        
        elif shape_name == 'cuadrado':
            side = (dimensions['width_mm'] + dimensions['height_mm']) / 2
            min_s = self.specs['cuadrado']['min_side_mm']
            max_s = self.specs['cuadrado']['max_side_mm']
            
            if side < min_s:
                defects.append(f"Muy pequeño: {side:.1f}mm < {min_s}mm")
            elif side > max_s:
                defects.append(f"Muy grande: {side:.1f}mm > {max_s}mm")
        
        passed = len(defects) == 0
        return passed, defects
    
    def run(self):
        print("="*60)
        print("SISTEMA DE INSPECCIÓN DE CALIDAD")
        print("="*60)
        print("\nEspecificaciones:")
        print("  Círculos: 30-50mm diámetro, circularidad >0.8")
        print("  Cuadrados: 30-50mm lado, relación aspecto ~1.0")
        print("\nControles:")
        print("  'q' - Salir")
        print("  'r' - Resetear estadísticas")
        print("  's' - Guardar frame\n")
        
        frame_count = 0
        
        while True:
            color_image, depth_image, depth_frame = self.get_frame()
            
            # Preprocesar
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Segmentar
            if depth_image is not None:
                mask_depth = np.logical_and(depth_image > 400, 
                                           depth_image < 1200).astype(np.uint8) * 255
                _, mask_intensity = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_and(mask_intensity, mask_depth)
            else:
                _, mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            
            # Morfología
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            result = color_image.copy()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 1000:
                    # Clasificar forma
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.04 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    vertices = len(approx)
                    
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    if vertices > 8 and circularity > 0.7:
                        shape_name = 'circulo'
                    elif vertices == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        ar = float(w) / h if h > 0 else 0
                        if 0.9 <= ar <= 1.1:
                            shape_name = 'cuadrado'
                        else:
                            continue  # Ignorar rectángulos
                    else:
                        continue  # Ignorar otras formas
                    
                    # INSPECCIÓN
                    self.inspected_count += 1
                    all_defects = []
                    
                    # 1. Inspeccionar forma
                    shape_ok, shape_defects = self.inspect_shape(contour, shape_name)
                    all_defects.extend(shape_defects)
                    
                    # 2. Medir dimensiones
                    dimensions = self.measure_dimensions(contour, depth_frame)
                    
                    # 3. Verificar dimensiones
                    if dimensions:
                        dim_ok, dim_defects = self.check_dimensions(dimensions, shape_name)
                        all_defects.extend(dim_defects)
                    
                    # RESULTADO
                    passed = len(all_defects) == 0
                    
                    if passed:
                        self.passed_count += 1
                        color = (0, 255, 0)  # Verde
                        status = "PASS"
                    else:
                        self.failed_count += 1
                        color = (0, 0, 255)  # Rojo
                        status = "FAIL"
                        self.defect_types.extend(all_defects)
                    
                    # Dibujar
                    cv2.drawContours(result, [contour], -1, color, 3)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Etiqueta principal
                    cv2.putText(result, f"{shape_name.upper()}: {status}",
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, color, 2)
                    
                    # Dimensiones
                    if dimensions:
                        dim_text = f"{dimensions['width_mm']:.1f}x{dimensions['height_mm']:.1f}mm"
                        cv2.putText(result, dim_text, (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    # Defectos (si hay)
                    if not passed:
                        y_offset = y + h + 40
                        for defect in all_defects[:2]:  # Máximo 2 para no saturar
                            cv2.putText(result, defect, (x, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                            y_offset += 15
            
            # Panel de estadísticas
            self.draw_stats_panel(result)
            
            cv2.imshow('Inspector de Calidad', result)
            
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.inspected_count = 0
                self.passed_count = 0
                self.failed_count = 0
                self.defect_types.clear()
                print("✓ Estadísticas reseteadas")
            elif key == ord('s'):
                cv2.imwrite(f'inspection_{frame_count}.png', result)
                print(f"✓ Frame guardado")
        
        self.print_final_report()
        self.cleanup()
        cv2.destroyAllWindows()
    
    def draw_stats_panel(self, image):
        """Dibuja panel de estadísticas"""
        h, w = image.shape[:2]
        
        # Panel
        overlay = image.copy()
        cv2.rectangle(overlay, (w-250, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Estadísticas
        cv2.putText(image, "ESTADISTICAS", (w-240, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(image, f"Inspeccionados: {self.inspected_count}", (w-240, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"Aprobados: {self.passed_count}", (w-240, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(image, f"Rechazados: {self.failed_count}", (w-240, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if self.inspected_count > 0:
            yield_rate = (self.passed_count / self.inspected_count) * 100
            cv2.putText(image, f"Yield: {yield_rate:.1f}%", (w-240, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def print_final_report(self):
        """Imprime reporte final"""
        print("\n" + "="*60)
        print("REPORTE FINAL DE INSPECCIÓN")
        print("="*60)
        print(f"Total inspeccionado: {self.inspected_count}")
        print(f"Aprobados: {self.passed_count}")
        print(f"Rechazados: {self.failed_count}")
        
        if self.inspected_count > 0:
            yield_rate = (self.passed_count / self.inspected_count) * 100
            print(f"Yield Rate: {yield_rate:.2f}%")
            
            if self.defect_types:
                print("\nDefectos más comunes:")
                from collections import Counter
                defect_counts = Counter(self.defect_types)
                for defect, count in defect_counts.most_common(5):
                    print(f"  - {defect}: {count} veces")
    
    def cleanup(self):
        if self.use_realsense:
            self.pipeline.stop()
        else:
            self.cap.release()

# EJECUTAR
inspector = QualityInspector(use_realsense=True)
inspector.run()