//! Common utilities for tests

pub mod core {
    use std::collections::HashMap;
    
    #[derive(Clone)]
    pub struct Config {
        options: HashMap<String, String>,
    }
    
    impl Config {
        pub fn new() -> Self {
            Config {
                options: HashMap::new(),
            }
        }
        
        pub fn set<T: ToString>(&mut self, key: &str, value: T) {
            self.options.insert(key.to_string(), value.to_string());
        }
        
        pub fn get<T: std::str::FromStr>(&self, key: &str) -> Option<T> {
            self.options.get(key).and_then(|value| {
                value.parse::<T>().ok()
            })
        }
    }
    
    pub struct SharedData<T: Clone> {
        data: T,
    }
    
    impl<T: Clone> SharedData<T> {
        pub fn new(data: T) -> Self {
            SharedData { data }
        }
        
        pub fn get(&self) -> T {
            self.data.clone()
        }
    }
    
    // Thread-local config for testing
    thread_local! {
        static THREAD_CONFIG: std::cell::RefCell<Option<Config>> = std::cell::RefCell::new(None);
    }
    
    pub fn with_config<F, R>(config: Config, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        THREAD_CONFIG.with(|cell| {
            *cell.borrow_mut() = Some(config);
        });
        
        let result = f();
        
        THREAD_CONFIG.with(|cell| {
            *cell.borrow_mut() = None;
        });
        
        result
    }
    
    pub fn get_config() -> Config {
        THREAD_CONFIG.with(|cell| {
            let borrowed = cell.borrow();
            match &*borrowed {
                Some(config) => config.clone(),
                None => Config::new()
            }
        })
    }
}

pub mod math {
    pub struct Vector {
        data: Vec<f64>,
    }
    
    impl Vector {
        pub fn new(data: Vec<f64>) -> Self {
            Vector { data }
        }
        
        pub fn data(&self) -> &[f64] {
            &self.data
        }
        
        pub fn add(&self, other: &Vector) -> Vector {
            let mut result = self.data.clone();
            for i in 0..result.len() {
                result[i] += other.data[i];
            }
            Vector::new(result)
        }
        
        pub fn sub(&self, other: &Vector) -> Vector {
            let mut result = self.data.clone();
            for i in 0..result.len() {
                result[i] -= other.data[i];
            }
            Vector::new(result)
        }
        
        pub fn mul(&self, scalar: f64) -> Vector {
            let mut result = self.data.clone();
            for i in 0..result.len() {
                result[i] *= scalar;
            }
            Vector::new(result)
        }
        
        pub fn dot(&self, other: &Vector) -> f64 {
            let mut result = 0.0;
            for i in 0..self.data.len() {
                result += self.data[i] * other.data[i];
            }
            result
        }
        
        pub fn norm(&self) -> f64 {
            f64::sqrt(self.dot(self))
        }
    }
    
    pub struct Matrix {
        data: Vec<Vec<f64>>,
    }
    
    impl Matrix {
        pub fn new(data: Vec<Vec<f64>>) -> Self {
            Matrix { data }
        }
        
        pub fn data(&self) -> &[Vec<f64>] {
            &self.data
        }
        
        pub fn add(&self, other: &Matrix) -> Matrix {
            let mut result = self.data.clone();
            for i in 0..result.len() {
                for j in 0..result[i].len() {
                    result[i][j] += other.data[i][j];
                }
            }
            Matrix::new(result)
        }
        
        pub fn sub(&self, other: &Matrix) -> Matrix {
            let mut result = self.data.clone();
            for i in 0..result.len() {
                for j in 0..result[i].len() {
                    result[i][j] -= other.data[i][j];
                }
            }
            Matrix::new(result)
        }
        
        pub fn mul_scalar(&self, scalar: f64) -> Matrix {
            let mut result = self.data.clone();
            for i in 0..result.len() {
                for j in 0..result[i].len() {
                    result[i][j] *= scalar;
                }
            }
            Matrix::new(result)
        }
        
        pub fn mul_matrix(&self, other: &Matrix) -> Matrix {
            let rows = self.data.len();
            let cols = other.data[0].len();
            let mut result = vec![vec![0.0; cols]; rows];
            
            for i in 0..rows {
                for j in 0..cols {
                    for k in 0..other.data.len() {
                        result[i][j] += self.data[i][k] * other.data[k][j];
                    }
                }
            }
            
            Matrix::new(result)
        }
        
        pub fn transpose(&self) -> Matrix {
            let rows = self.data.len();
            let cols = self.data[0].len();
            let mut result = vec![vec![0.0; rows]; cols];
            
            for i in 0..rows {
                for j in 0..cols {
                    result[j][i] = self.data[i][j];
                }
            }
            
            Matrix::new(result)
        }
    }
    
    pub mod stats {
        pub fn sum(data: &[f64]) -> f64 {
            if data.is_empty() {
                return 0.0;
            }
            data.iter().sum()
        }
        
        pub fn mean(data: &[f64]) -> f64 {
            if data.is_empty() {
                return 0.0;
            }
            sum(data) / data.len() as f64
        }
        
        pub fn median(data: &[f64]) -> f64 {
            if data.is_empty() {
                return 0.0;
            }
            
            let mut sorted = data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
        
        pub fn variance(data: &[f64]) -> f64 {
            if data.len() < 2 {
                return 0.0;
            }
            
            let m = mean(data);
            let mut sum_squared_diff = 0.0;
            
            for &x in data {
                let diff = x - m;
                sum_squared_diff += diff * diff;
            }
            
            sum_squared_diff / data.len() as f64
        }
        
        pub fn std(data: &[f64]) -> f64 {
            f64::sqrt(variance(data))
        }
        
        pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
            if x.len() != y.len() || x.len() < 2 {
                return 0.0;
            }
            
            let mean_x = mean(x);
            let mean_y = mean(y);
            
            let mut numerator = 0.0;
            let mut sum_sq_x = 0.0;
            let mut sum_sq_y = 0.0;
            
            for i in 0..x.len() {
                let diff_x = x[i] - mean_x;
                let diff_y = y[i] - mean_y;
                
                numerator += diff_x * diff_y;
                sum_sq_x += diff_x * diff_x;
                sum_sq_y += diff_y * diff_y;
            }
            
            if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
                return 0.0;
            }
            
            numerator / f64::sqrt(sum_sq_x * sum_sq_y)
        }
    }
}

pub mod text {
    pub fn reverse(s: &str) -> String {
        s.chars().rev().collect()
    }
    
    pub fn base64_encode(s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }
        
        use base64::{Engine as _, engine::general_purpose};
        general_purpose::STANDARD.encode(s.as_bytes())
    }
    
    pub fn base64_decode(s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }
        
        use base64::{Engine as _, engine::general_purpose};
        let bytes = general_purpose::STANDARD.decode(s.as_bytes()).unwrap_or_default();
        String::from_utf8(bytes).unwrap_or_default()
    }
    
    pub fn split<'a>(s: &'a str, delimiter: &str) -> Vec<&'a str> {
        s.split(delimiter).collect()
    }
    
    pub fn join(parts: &[&str], delimiter: &str) -> String {
        parts.join(delimiter)
    }
    
    pub fn replace(s: &str, from: &str, to: &str) -> String {
        s.replace(from, to)
    }
    
    pub fn regex_replace(s: &str, pattern: &str, replacement: &str) -> String {
        use regex::Regex;
        match Regex::new(pattern) {
            Ok(re) => re.replace_all(s, replacement).to_string(),
            Err(_) => s.to_string(),
        }
    }
    
    pub fn to_uppercase(s: &str) -> String {
        s.to_uppercase()
    }
    
    pub fn to_lowercase(s: &str) -> String {
        s.to_lowercase()
    }
    
    pub fn tokenize(s: &str) -> Vec<String> {
        tokenize_with_options(s, true, true)
    }
    
    pub fn tokenize_with_options(s: &str, lowercase: bool, remove_punct: bool) -> Vec<String> {
        if s.is_empty() {
            return Vec::new();
        }
        
        let mut text = s.to_string();
        
        if lowercase {
            text = text.to_lowercase();
        }
        
        if remove_punct {
            text = text.chars()
                .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
                .collect();
        }
        
        text.split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
    
    pub fn ngrams(s: &str, n: usize) -> Vec<Vec<String>> {
        let tokens: Vec<String> = s.split_whitespace().map(|s| s.to_string()).collect();
        let tokens_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        ngrams_from_tokens(&tokens_refs, n)
    }
    
    pub fn ngrams_from_tokens(tokens: &[&str], n: usize) -> Vec<Vec<String>> {
        if tokens.len() < n {
            return Vec::new();
        }
        
        let mut result = Vec::new();
        for i in 0..=tokens.len() - n {
            let ngram: Vec<String> = tokens[i..i+n].iter().map(|&s| s.to_string()).collect();
            result.push(ngram);
        }
        
        result
    }
}

pub mod data {
    use std::collections::HashMap;
    
    pub fn filter<T: Clone>(items: &[T], predicate: impl Fn(&T) -> bool) -> Vec<T> {
        items.iter().filter(|item| predicate(item)).cloned().collect()
    }
    
    pub fn map<T: Clone, U>(items: &[T], f: impl Fn(&T) -> U) -> Vec<U> {
        items.iter().map(f).collect()
    }
    
    pub fn reduce<T: Clone, U>(items: &[T], f: impl Fn(U, &T) -> U, initial: U) -> U {
        items.iter().fold(initial, f)
    }
    
    pub fn sort<T: Clone + Ord>(items: &mut [T], reverse: bool) {
        if reverse {
            items.sort_by(|a, b| b.cmp(a));
        } else {
            items.sort();
        }
    }
    
    pub struct DataFrame {
        data: HashMap<String, Vec<String>>,
    }
    
    impl DataFrame {
        pub fn new(data: HashMap<String, Vec<String>>) -> Self {
            DataFrame { data }
        }
        
        pub fn get_column(&self, name: &str) -> Option<&Vec<String>> {
            self.data.get(name)
        }
        
        pub fn num_rows(&self) -> usize {
            self.data.values().next().map_or(0, |v| v.len())
        }
        
        pub fn column_names(&self) -> Vec<String> {
            self.data.keys().cloned().collect()
        }
        
        pub fn apply<F>(&self, f: F) -> DataFrame
        where
            F: Fn(&Vec<String>) -> Vec<String>
        {
            let mut result = HashMap::new();
            for (name, column) in &self.data {
                result.insert(name.clone(), f(column));
            }
            DataFrame::new(result)
        }
        
        pub fn groupby(&self, by_column: &str, aggregations: HashMap<String, String>) -> DataFrame {
            let by_values = match self.get_column(by_column) {
                Some(col) => col,
                None => return DataFrame::new(HashMap::new()),
            };
            
            // Group by unique values
            let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
            for (i, value) in by_values.iter().enumerate() {
                groups.entry(value.clone()).or_default().push(i);
            }
            
            // Create result dataframe
            let mut result_data = HashMap::new();
            
            // Add the groupby column
            let group_keys: Vec<String> = groups.keys().cloned().collect();
            result_data.insert(by_column.to_string(), group_keys.clone());
            
            // Apply aggregations
            for (col_name, agg_type) in aggregations {
                if let Some(column) = self.get_column(&col_name) {
                    let mut agg_result = Vec::with_capacity(groups.len());
                    
                    for group_key in &group_keys {
                        let indices = &groups[group_key];
                        let group_values: Vec<f64> = indices.iter()
                            .filter_map(|&i| column[i].parse::<f64>().ok())
                            .collect();
                        
                        let agg_value = match agg_type.as_str() {
                            "mean" => {
                                if group_values.is_empty() {
                                    0.0
                                } else {
                                    group_values.iter().sum::<f64>() / group_values.len() as f64
                                }
                            },
                            "sum" => group_values.iter().sum::<f64>(),
                            "count" => indices.len() as f64,
                            "min" => group_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                            "max" => group_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                            _ => 0.0,
                        };
                        
                        agg_result.push(agg_value.to_string());
                    }
                    
                    result_data.insert(format!("{}_{}", col_name, agg_type), agg_result);
                }
            }
            
            DataFrame::new(result_data)
        }
    }
}

pub mod io {
    use std::fs::File;
    use std::io::{self, Read, Write};
    use std::path::Path;
    
    pub fn read_file(path: &str) -> io::Result<String> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    }
    
    pub fn write_file(path: &str, contents: &str) -> io::Result<()> {
        // Create parent directories if they don't exist
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let mut file = File::create(path)?;
        file.write_all(contents.as_bytes())?;
        Ok(())
    }
    
    pub fn append_file(path: &str, contents: &str) -> io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        
        file.write_all(contents.as_bytes())?;
        Ok(())
    }
    
    pub fn file_exists(path: &str) -> bool {
        Path::new(path).exists()
    }
    
    pub fn delete_file(path: &str) -> io::Result<()> {
        std::fs::remove_file(path)
    }
}

pub mod image {
    #[derive(Clone)]
    pub struct Image {
        width: usize,
        height: usize,
        pixels: Vec<u8>,  // RGBA format
    }
    
    impl Image {
        pub fn new(width: usize, height: usize) -> Self {
            let pixels = vec![0; width * height * 4];
            Image { width, height, pixels }
        }
        
        pub fn width(&self) -> usize {
            self.width
        }
        
        pub fn height(&self) -> usize {
            self.height
        }
        
        pub fn get_pixel(&self, x: usize, y: usize) -> [u8; 4] {
            if x >= self.width || y >= self.height {
                return [0, 0, 0, 0];
            }
            
            let idx = (y * self.width + x) * 4;
            [
                self.pixels[idx],
                self.pixels[idx + 1],
                self.pixels[idx + 2],
                self.pixels[idx + 3],
            ]
        }
        
        pub fn set_pixel(&mut self, x: usize, y: usize, rgba: [u8; 4]) {
            if x >= self.width || y >= self.height {
                return;
            }
            
            let idx = (y * self.width + x) * 4;
            self.pixels[idx] = rgba[0];
            self.pixels[idx + 1] = rgba[1];
            self.pixels[idx + 2] = rgba[2];
            self.pixels[idx + 3] = rgba[3];
        }
        
        pub fn resize(&self, new_width: usize, new_height: usize) -> Image {
            let mut result = Image::new(new_width, new_height);
            
            let x_ratio = self.width as f64 / new_width as f64;
            let y_ratio = self.height as f64 / new_height as f64;
            
            for y in 0..new_height {
                for x in 0..new_width {
                    let px = (x as f64 * x_ratio) as usize;
                    let py = (y as f64 * y_ratio) as usize;
                    let pixel = self.get_pixel(px, py);
                    result.set_pixel(x, y, pixel);
                }
            }
            
            result
        }
        
        pub fn grayscale(&self) -> Image {
            let mut result = Image::new(self.width, self.height);
            
            for y in 0..self.height {
                for x in 0..self.width {
                    let [r, g, b, a] = self.get_pixel(x, y);
                    let gray = (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) as u8;
                    result.set_pixel(x, y, [gray, gray, gray, a]);
                }
            }
            
            result
        }
        
        pub fn blur(&self, radius: usize) -> Image {
            let mut result = Image::new(self.width, self.height);
            
            for y in 0..self.height {
                for x in 0..self.width {
                    let mut r_sum = 0;
                    let mut g_sum = 0;
                    let mut b_sum = 0;
                    let mut a_sum = 0;
                    let mut count = 0;
                    
                    let start_y = y.saturating_sub(radius);
                    let end_y = (y + radius + 1).min(self.height);
                    let start_x = x.saturating_sub(radius);
                    let end_x = (x + radius + 1).min(self.width);
                    
                    for py in start_y..end_y {
                        for px in start_x..end_x {
                            let [r, g, b, a] = self.get_pixel(px, py);
                            r_sum += r as u32;
                            g_sum += g as u32;
                            b_sum += b as u32;
                            a_sum += a as u32;
                            count += 1;
                        }
                    }
                    
                    result.set_pixel(x, y, [
                        (r_sum / count) as u8,
                        (g_sum / count) as u8,
                        (b_sum / count) as u8,
                        (a_sum / count) as u8,
                    ]);
                }
            }
            
            result
        }
    }
}

pub mod ml {
    use rand::prelude::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    pub struct KMeansResult {
        pub centroids: Vec<Vec<f64>>,
        pub clusters: Vec<Vec<usize>>,
    }

    pub struct LinearRegressionResult {
        pub coefficients: Vec<f64>,
        pub intercept: f64,
        pub r_squared: f64,
    }

    pub fn kmeans(data: &[Vec<f64>], k: usize, max_iterations: usize, seed: u64) -> KMeansResult {
        if data.is_empty() {
            return KMeansResult {
                centroids: Vec::new(),
                clusters: Vec::new(),
            };
        }

        // Initialize centroids randomly
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centroids = Vec::with_capacity(k);
        let mut used_indices = HashSet::new();
        
        while centroids.len() < k {
            let idx = (rng.gen::<f64>() * data.len() as f64) as usize % data.len();
            if used_indices.insert(idx) {
                centroids.push(data[idx].clone());
            }
        }
        
        // Run k-means algorithm
        let mut cluster_assignments = vec![0; data.len()];
        let mut iterations = 0;
        let mut changed = true;
        
        while changed && iterations < max_iterations {
            changed = false;
            iterations += 1;
            
            // Assign points to clusters
            for (i, point) in data.iter().enumerate() {
                let mut min_dist = f64::MAX;
                let mut min_cluster = 0;
                
                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        min_cluster = j;
                    }
                }
                
                if cluster_assignments[i] != min_cluster {
                    cluster_assignments[i] = min_cluster;
                    changed = true;
                }
            }
            
            if !changed {
                break;
            }
            
            // Update centroids
            let dimensions = data[0].len();
            let mut new_centroids = vec![vec![0.0; dimensions]; k];
            let mut counts = vec![0; k];
            
            for (i, point) in data.iter().enumerate() {
                let cluster = cluster_assignments[i];
                counts[cluster] += 1;
                
                for j in 0..dimensions {
                    new_centroids[cluster][j] += point[j];
                }
            }
            
            for i in 0..k {
                if counts[i] > 0 {
                    for j in 0..dimensions {
                        new_centroids[i][j] /= counts[i] as f64;
                    }
                    centroids[i] = new_centroids[i].clone();
                }
            }
        }
        
        // Convert cluster assignments to clusters
        let mut clusters = vec![Vec::new(); k];
        for (i, &cluster) in cluster_assignments.iter().enumerate() {
            clusters[cluster].push(i);
        }
        
        KMeansResult {
            centroids,
            clusters,
        }
    }

    pub fn linear_regression(x: &[f64], y: &[f64]) -> LinearRegressionResult {
        if x.len() != y.len() || x.len() < 2 {
            return LinearRegressionResult {
                coefficients: Vec::new(),
                intercept: 0.0,
                r_squared: 0.0,
            };
        }
        
        // Calculate means
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        // Calculate slope and intercept
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..x.len() {
            let x_diff = x[i] - mean_x;
            let y_diff = y[i] - mean_y;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }
        
        let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
        let intercept = mean_y - slope * mean_x;
        
        // Calculate R-squared
        let mut ss_total = 0.0;
        let mut ss_residual = 0.0;
        
        for i in 0..x.len() {
            let y_pred = slope * x[i] + intercept;
            ss_total += (y[i] - mean_y).powi(2);
            ss_residual += (y[i] - y_pred).powi(2);
        }
        
        let r_squared = if ss_total != 0.0 { 1.0 - ss_residual / ss_total } else { 0.0 };
        
        LinearRegressionResult {
            coefficients: vec![slope],
            intercept,
            r_squared,
        }
    }

    pub fn normalize(data: &[Vec<f64>], method: &str) -> Vec<Vec<f64>> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let dimensions = data[0].len();
        let mut result = vec![vec![0.0; dimensions]; data.len()];
        
        for j in 0..dimensions {
            // Extract column
            let mut column = Vec::with_capacity(data.len());
            for i in 0..data.len() {
                column.push(data[i][j]);
            }
            
            // Normalize column
            let normalized = match method {
                "min-max" => {
                    let min = column.iter().fold(f64::MAX, |a, &b| a.min(b));
                    let max = column.iter().fold(f64::MIN, |a, &b| a.max(b));
                    let range = max - min;
                    
                    if range == 0.0 {
                        vec![0.5; column.len()]
                    } else {
                        column.iter().map(|&x| (x - min) / range).collect()
                    }
                },
                "z-score" => {
                    let mean = column.iter().sum::<f64>() / column.len() as f64;
                    let variance = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
                    let std_dev = variance.sqrt();
                    
                    if std_dev == 0.0 {
                        vec![0.0; column.len()]
                    } else {
                        column.iter().map(|&x| (x - mean) / std_dev).collect()
                    }
                },
                _ => column,  // Return original column for unknown method
            };
            
            // Put normalized column back into result
            for i in 0..data.len() {
                result[i][j] = normalized[i];
            }
        }
        
        result
    }

    pub fn distance_matrix(points: &[Vec<f64>], metric: &str) -> Vec<Vec<f64>> {
        if points.is_empty() {
            return Vec::new();
        }
        
        let n = points.len();
        let mut result = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                result[i][j] = match metric {
                    "euclidean" => euclidean_distance(&points[i], &points[j]),
                    "manhattan" => {
                        let mut sum = 0.0;
                        for k in 0..points[i].len() {
                            sum += (points[i][k] - points[j][k]).abs();
                        }
                        sum
                    },
                    _ => euclidean_distance(&points[i], &points[j]),  // Default to Euclidean
                };
            }
        }
        
        result
    }

    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }
}