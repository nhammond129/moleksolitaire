extern crate opencv;
use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};


#[derive(Debug, Clone)]
struct Card {
    name: &'static str,
    image: core::Mat,
}

impl Card {
    fn new(name: &'static str) -> Card {
        Card {
            name: name,
            image: imgcodecs::imread(&format!("cards/{}.png", name), imgcodecs::IMREAD_GRAYSCALE)
                .expect(&format!("Failed to load card {}", name))
        }
    }

    fn all() -> Vec<Card> {
        ["6", "7", "8", "9", "10", "V", "D", "K", "T"]
            .iter()
            .map(|name| Card::new(name))
            .collect()
    }
}

impl Into<u8> for &Card {
    fn into(self) -> u8 {
        match self.name {
            "6"  =>  6,
            "7"  =>  7,
            "8"  =>  8,
            "9"  =>  9,
            "10" => 10,
            "V"  => 11,
            "D"  => 12,
            "K"  => 13,
            "T"  => 14,
            _ => panic!("Invalid card name")
        }
    }
}

impl Into<u8> for Card {
    fn into(self) -> u8 {
        (&self).into()
    }
}

impl Into<String> for &Card {
    fn into(self) -> String {
        self.name.to_string()
    }
}

impl From<u8> for Card {
    fn from(value: u8) -> Self {
        match value {
            6  => Card::new("6"),
            7  => Card::new("7"),
            8  => Card::new("8"),
            9  => Card::new("9"),
            10 => Card::new("10"),
            11 => Card::new("V"),
            12 => Card::new("D"),
            13 => Card::new("K"),
            14 => Card::new("T"),
            _ => panic!("Invalid card value")
        }
    }
}

impl From<&str> for Card {
    fn from(value: &str) -> Self {
        match value {
            "6"  => Card::new("6"),
            "7"  => Card::new("7"),
            "8"  => Card::new("8"),
            "9"  => Card::new("9"),
            "10" => Card::new("10"),
            "V"  => Card::new("V"),
            "D"  => Card::new("D"),
            "K"  => Card::new("K"),
            "T"  => Card::new("T"),
            _ => panic!("Invalid card value")
        }
    }
}

struct Board {
    columns: [Vec<u8>; 6]
}

impl From<Vec<u8>> for Board {
    fn from(values: Vec<u8>) -> Self {
        let mut columns: [Vec<u8>; 6] = Default::default();
        for i in 0..6 {
            columns[i] = Vec::new();
        }

        // push value i into column i%6
        for (i, value) in values.iter().enumerate() {
            match value {
                6..=14 => columns[i % 6].push(*value),
                _ => continue  // skip values outside of 6 to 14
            }
        }

        Board {
            columns: columns
        }
    }
}

impl Board {
    fn from_image(image: &core::Mat) -> Board {
        let mut columns: [Vec<u8>; 6] = Default::default();
        for i in 0..6 {
            columns[i] = Vec::new();
        }

        let all_cards = Card::all();

        struct CardMatch {
            card: Card,
            pos: (i32, i32)
        }

        let mut matches: Vec<CardMatch> = Vec::new();

        for card in all_cards.iter() {
            let mut result = core::Mat::default();
            let _ = imgproc::match_template(&image, &card.image, &mut result, imgproc::TM_CCOEFF_NORMED, &core::no_array());

            // take high-confidence matches
            let mut thresholded = core::Mat::default();
            let _ = imgproc::threshold(&result, &mut thresholded, 0.9, 1.0, imgproc::THRESH_BINARY);

            let mut locations = core::Mat::default();
            let _ = core::find_non_zero(&thresholded, &mut locations);

            for i in 0..locations.rows() {
                let xy: &core::Point = locations.at::<core::Point>(i).unwrap();
                matches.push(CardMatch { card: card.clone(), pos: (xy.x, xy.y) });
            }
        }

        matches.sort_by_key(|m| m.pos.1);
        matches.truncate(36);
        matches.sort_by_key(|m| m.pos.0 + 20*m.pos.1);

        // let mut img_copy = core::Mat::default();
        // let _ = imgproc::cvt_color(&image, &mut img_copy, imgproc::COLOR_GRAY2RGB, 0);
        // insert card i into column i%6
        for (i, match_) in matches.iter().enumerate() {
            // const GREEN: core::Scalar = core::Scalar::new(0.0, 255.0, 0.0, 0.0);
            // const RED: core::Scalar = core::Scalar::new(0.0, 0.0, 255.0, 0.0);
            // let _ = imgproc::put_text(&mut img_copy, match_.card.name, core::Point::new(match_.pos.0, match_.pos.1+24), imgproc::FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2, 8, false);
            // let _ = imgproc::put_text(&mut img_copy, &format!("{}", i), core::Point::new(match_.pos.0+40, match_.pos.1+16), imgproc::FONT_HERSHEY_SIMPLEX, 0.75, RED, 2, 8, false);
            columns[i % 6].push((&match_.card).into());
        }

        // highgui::imshow("Matches", &img_copy).expect("Failed to show matches");
        // loop {
        //     if highgui::wait_key(0).unwrap() == 27 {
        //         break;
        //     }
        // }


        Board {
            columns: columns
        }
    }

    fn debug_print(&self) {
        let mut rows: Vec<Vec<u8>> = Vec::new();

        let mut depth = 0;
        let maxdepth = self.columns.iter().map(|column| column.len()).max().unwrap();
        while depth < maxdepth {
            let mut row: Vec<u8> = Vec::new();
            for column in self.columns.iter() {
                if depth < column.len() {
                    row.push(column[depth]);
                } else {
                    row.push(0);
                }
            }
            rows.push(row);
            depth += 1;
        }

        rows.iter().for_each(|row| {
            row.iter().for_each(|value| {
                match value {
                    0 => print!("    "),
                    14 => print!("  T "),
                    13 => print!("  K "),
                    12 => print!("  D "),
                    11 => print!("  V "),
                    _ => print!(" {:2} ", value)
                }
            });
            println!();
        });
    }
}

fn main() {
    // screenshot desktop
    // make sure it's a fresh board
    // solve
    // run clicks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_cards() {
        let cards = Card::all();
        assert_eq!(cards.len(), 9);

        for (card, expected_name) in cards.iter().zip(["6", "7", "8", "9", "10", "V", "D", "K", "T"].iter()) {
            assert_eq!(&card.name, expected_name);
        }
    }

    #[test]
    fn read_cropped_board() {
        let image = imgcodecs::imread("testdata/cropped.png", imgcodecs::IMREAD_GRAYSCALE).expect("Failed to load board image");
        let board = Board::from_image(&image);

        board.columns.iter().for_each(|column| {
            assert_eq!(column.len(), 6);
        });

        let expected_board = Board::from(vec![
            11, 10, 14, 12, 14, 11,
             9,  8,  8, 13, 12,  6,
            10,  6,  6,  9,  8, 13,
            12, 11, 13,  9,  9, 13,
            12,  7, 14, 10, 10,  6,
             7,  7,  7, 14,  8, 11,
        ]);


        println!("Got:");
        board.debug_print();
    
        println!("Expected:");
        expected_board.debug_print();

        board.columns.iter().zip(expected_board.columns.iter()).for_each(|(column, expected_column)| {
            column.iter().zip(expected_column.iter()).for_each(|(value, expected_value)| {
                assert_eq!(value, expected_value);
            });
        });

    }
}